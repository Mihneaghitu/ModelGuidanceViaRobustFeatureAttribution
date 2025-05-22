import torch
import tqdm
from math import ceil, floor
import sys
from .robust_regularizer import (input_gradient_interval_regularizer,
                                 input_gradient_pgd_regularizer,
                                 smooth_gradient_regularizer,
                                 llm_gradient_regularizer)
from models.llm import LLM, Tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import yaml

def train_model_with_certified_input_grad(
    dl_train: torch.utils.data.DataLoader,
    n_epochs: int,
    model: torch.nn.Module,
    learning_rate: float,
    criterion: torch.nn.Module,
    epsilon: float,
    mlx_method: str, # one of ["std", "grad_cert", "ibp_ex", "r3" or "r4"]
    k: float, # input reg weight
    device: str,
    has_conv: bool,
    k_schedule: callable = None,
    weight_reg_coeff: float = 0.0,
    class_weights: list[float] = None,
    weight_decay: float = 0.0,
    suppress_tqdm: bool = False
) -> None:
    loss_fn = None
    if isinstance(criterion, torch.nn.BCELoss):
        loss_fn = "binary_cross_entropy"
    elif isinstance(criterion, torch.nn.CrossEntropyLoss):
        loss_fn = "cross_entropy"
    else:
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # pre-train the model
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", ) if not suppress_tqdm else range(n_epochs)
    model = model.to(device)
    for curr_epoch in progress_bar:
        for i, (x, u, m) in enumerate(dl_train):
            # Forward pass
            u, x, m = u.to(device), x.to(device), m.to(device)

            # For std, we will waste some time doing the bounds, but at least it is consistent across methods
            inp_grad_reg = input_gradient_interval_regularizer(
                model, x, u, loss_fn, epsilon, 0.0, regularizer_type=mlx_method, batch_masks=m, has_conv=has_conv,
                device=device, weight_reg_coeff=weight_reg_coeff
            )
            if mlx_method == "std":
                assert inp_grad_reg == 0
                if loss_fn == "cross_entropy":
                # The last module is either Softmax or Sigmoid, hence why the [-2] is used
                    u = torch.nn.functional.one_hot(u, num_classes=list(model.modules())[-2].out_features).float()
            # output is [batch_size, 1], u is [bach_size] but BCELoss does not support different target and source sizes
            output = model(x).squeeze()
            # for unbalanced plant
            if class_weights is not None and isinstance(criterion, torch.nn.BCELoss):
                batch_weights = torch.tensor([class_weights[int(label.item())] for label in u]).to(device)
                criterion = torch.nn.BCELoss(weight=batch_weights)
            std_loss = criterion(output, u)
            if k_schedule is not None:
                k = k_schedule(curr_epoch, n_epochs)
            loss = std_loss + k * inp_grad_reg
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                if not suppress_tqdm:
                    progress_bar.set_postfix({"loss (entropy)": loss.item(), "reg": k * inp_grad_reg})
                else:
                    print(f"Epoch {curr_epoch}, loss (entropy): {loss.item()}, reg: {k * inp_grad_reg}")

def train_model_with_pgd_robust_input_grad(
    dl_train: torch.utils.data.DataLoader,
    n_epochs: int,
    model: torch.nn.Module,
    learning_rate: float,
    criterion: torch.nn.Module,
    epsilon: float,
    mlx_method: str, # one of ["std", "grad_cert", "ibp_ex", "r3" or "r4"]
    k: float, # input reg weight
    device: str,
    weight_reg_coeff: float = 0.0,
    class_weights: list[float] = None,
    num_iterations: int = 10,
    weight_decay: float = 0.0,
    k_schedule: callable = None,
    suppress_tqdm: bool = False
) -> None:
    if not (isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.CrossEntropyLoss)):
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # pre-train the model
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", ) if not suppress_tqdm else range(n_epochs)
    model = model.to(device).train()
    for curr_epoch in progress_bar:
        for i, (x, u, m) in enumerate(dl_train):
            # if the batch is not full (last batch), skip it
            if x.shape[0] != dl_train.batch_size:
                continue
            # Forward pass
            u, x, m = u.to(device), x.to(device), m.to(device)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                # The last module is either Softmax or Sigmoid, hence why the [-2] is used
                u = torch.nn.functional.one_hot(u, num_classes=list(model.modules())[-2].out_features).float()
            inp_grad_reg = input_gradient_pgd_regularizer(
                x, u, model, m, criterion, epsilon, num_iterations=num_iterations, regularizer_type=mlx_method, device=device, weight_reg_coeff=weight_reg_coeff
            )
            if mlx_method == "std":
                assert inp_grad_reg == 0
            # output is [batch_size, 1], u is [bach_size] but BCELoss does not support different target and source sizes
            output = model(x).squeeze()
            if class_weights is not None and isinstance(criterion, torch.nn.BCELoss):
                batch_weights = torch.tensor([class_weights[int(label.item())] for label in u]).to(device)
                criterion = torch.nn.BCELoss(weight=batch_weights)
            std_loss = criterion(output, u)
            alpha = 1
            if k_schedule is not None:
                alpha = k_schedule(curr_epoch, n_epochs, std_loss.item(), inp_grad_reg)
            loss = std_loss + alpha * k * inp_grad_reg
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not suppress_tqdm:
                if i % 100 == 0:
                    progress_bar.set_postfix({"loss": loss.item(), "reg": k * inp_grad_reg})


def train_model_with_smoothed_input_grad(
    dl_train: torch.utils.data.DataLoader,
    n_epochs: int,
    model: torch.nn.Module,
    learning_rate: float,
    criterion: torch.nn.Module,
    epsilon: float,
    mlx_method: str, # one of ["rand_r4, "smooth_r3"]
    k: float,
    device: str,
    num_samples: int = 10,
    weight_reg_coeff: float = 0.0,
    weight_decay: float = 0.0,
    class_weights: list[float] = None,
    suppress_tqdm: bool = False
) -> None:
    if not (isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.CrossEntropyLoss)):
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", ) if not suppress_tqdm else range(n_epochs)
    model = model.to(device).train()
    for _ in progress_bar:
        for i, (x, u, m) in enumerate(dl_train):
            # if the batch is not full (last batch), skip it
            if x.shape[0] != dl_train.batch_size:
                continue
            # Forward pass
            u, x, m = u.to(device), x.to(device), m.to(device)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                # The last module is either Softmax or Sigmoid, hence why the [-2] is used
                u = torch.nn.functional.one_hot(u, num_classes=list(model.modules())[-2].out_features).float()
            # For std, we will waste some time doing the bounds, but at least it is consistent across methods
            inp_grad_reg = smooth_gradient_regularizer(
                x, u, model, m, criterion, epsilon, regularizer_type=mlx_method, device=device, num_samples=num_samples, weight_reg_coeff=weight_reg_coeff)
            # output is [batch_size, 1], u is [bach_size] but BCELoss does not support different target and source sizes
            output = model(x).squeeze()
            if class_weights is not None and isinstance(criterion, torch.nn.BCELoss):
                batch_weights = torch.tensor([class_weights[int(label.item())] for label in u]).to(device)
                criterion = torch.nn.BCELoss(weight=batch_weights)
            std_loss = criterion(output, u)
            loss = std_loss + k * inp_grad_reg
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not suppress_tqdm:
                if i % 100 == 0:
                    progress_bar.set_postfix({"loss": loss.item(), "reg": k * inp_grad_reg})

def train_llm_with_guidance(
    model: LLM,
    tokenizer: Tokenizer,
    dl_train: torch.utils.data.DataLoader,
    n_epochs: int,
    learning_rate: float,
    criterion: torch.nn.Module,
	# Note: "pgd_r4" is going to actually be GCG for LLMs
    mlx_method: str, # one of [r3, "smooth_r3", "rand_r4", "pgd_r4"]
    lmbda: float,
    device: str,
    num_fragments: int = 1,
    alpha: float = 0.1,
    num_samples: int = 5,
    weight_decay: float = 0.0,
    class_weights: list[float] = None,
    suppress_tqdm: bool = False
) -> None:
    if not (isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.CrossEntropyLoss)):
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", ) if not suppress_tqdm else range(n_epochs)
    model = model.to(device).train()
    class_weights = torch.tensor(class_weights).to(device) if class_weights is not None else None
    # with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        for _ in progress_bar:
            for i, (text, labels, masks) in enumerate(dl_train):
                # Skip incomplete batches because fragmentation logic gets quite complicated quiclkly
                if labels.shape[0] != dl_train.batch_size:
                    continue
                model.zero_grad()
                batch_std_loss, batch_reg_loss = 0, 0
                # fragment the batch into num_fragments
                if num_fragments == 1:
                    fragmented_text, labels, fragmented_masks = [text], [labels], [masks]
                else:
                    labels = list(torch.chunk(labels, num_fragments))
                    # These are list of strings, so they need to be handled separately
                    fragmented_text, fragmented_masks = [], []
                    frag_size = int(ceil(len(text) / num_fragments))
                    for fragment_idx in range(num_fragments):
                        if fragment_idx < num_fragments - 1:
                            fragmented_text.append(text[fragment_idx * frag_size:(fragment_idx + 1) * frag_size])
                            fragmented_masks.append(masks[fragment_idx * frag_size:(fragment_idx + 1) * frag_size])
                        else:
                            fragmented_text.append(text[fragment_idx * frag_size:])
                            fragmented_masks.append(masks[fragment_idx * frag_size:])
                for fragment_idx in range(num_fragments):
                    text_fragment, labels_fragment, masks_fragment = fragmented_text[fragment_idx], labels[fragment_idx], fragmented_masks[fragment_idx]
                    encoding_fragment = tokenizer.tokenize(text_fragment)
                    token_ids_fragment, attention_mask_fragment = encoding_fragment['input_ids'], encoding_fragment['attention_mask']
                    # Forward pass
                    token_ids_fragment, attention_mask_fragment, labels_fragment = token_ids_fragment.to(device), attention_mask_fragment.to(device), labels_fragment.to(device)
                    # Compute gradient with respect to the masked area and regularize based on the method
                    inp_grad_reg = llm_gradient_regularizer(
                        model,
                        tokenizer,
                        text_fragment,
                        labels_fragment,
                        masks_fragment,
                        criterion,
                        alpha=alpha,
                        num_samples=num_samples,
                        regularizer_type=mlx_method,
                        device=device
                    )
                    output_fragment = model(token_ids_fragment, attention_mask_fragment).squeeze(-1)
                    if class_weights is not None and isinstance(criterion, torch.nn.BCELoss):
                        criterion = torch.nn.BCELoss(weight=class_weights[labels_fragment.int()].to(device))
                    std_loss = criterion(output_fragment, labels_fragment)
                    fragment_loss = std_loss + lmbda * inp_grad_reg
                    fragment_loss.backward()
                    batch_std_loss += std_loss.cpu().float().item()
                    batch_reg_loss += inp_grad_reg.cpu().float().item()
                    torch.cuda.empty_cache()
                # Backward and optimize
                optimizer.step()
                optimizer.zero_grad()
                if not suppress_tqdm:
                    if i % 25 == 0:
                        progress_bar.set_postfix({"loss": batch_std_loss / num_fragments, "reg": lmbda * batch_reg_loss / num_fragments})

@torch.no_grad()
def test_llm_accuracy(
    model: LLM,
    tokenizer: Tokenizer,
    dl_test: torch.utils.data.DataLoader,
    device: str,
    multi_class: bool = False,
    suppress_log: bool = False
) -> float:
    n_corr, n = 0, 0
    for text, labels, *_ in dl_test:
        # Forward pass
        labels = labels.to(device)
        encoding = tokenizer.tokenize(text)
        tokens, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)
        output = model(tokens, attention_mask).squeeze(-1)
        if multi_class:
            predicted_labels = output.argmax(dim=-1).squeeze()
        else:
            predicted_labels = (output > 0.5).int().squeeze()
        batch_correct = (predicted_labels == labels).sum().item()
        n_corr += batch_correct
        n += labels.shape[0]
    all_acc = n_corr / n
    if not suppress_log:
        print(f"Accuracy = {all_acc:.4g}")

    return round(all_acc, 4)

def test_model_accuracy(model: torch.nn.Sequential, dl_test: torch.utils.data.DataLoader, device: str,
                        multi_class: bool = False, suppress_log: bool = False) -> float:
    all_acc = 0
    num_inputs = 0
    model.eval()
    for test_batch, test_labels, *_ in dl_test: # when groups are present, _ is a tuple
        # Just do a simple forward and compare the output to the labels
        test_batch, test_labels = test_batch.to(device), test_labels.to(device)
        output = model(test_batch).squeeze()
        correct = 0
        if multi_class:
            correct = (output.argmax(dim=1) == test_labels).sum().item()
        else:
            correct = ((output > 0.5) == (test_labels)).sum().item()
        all_acc += correct
        num_inputs += test_batch.shape[0]
    all_acc /= num_inputs
    if not suppress_log:
        print("--- Model accuracy ---")
        print(f"Nominal = {all_acc:.4g}")

    return round(all_acc, 4)

def test_macro_avg_label_accuracy(model: torch.nn.Sequential, dl_test: torch.utils.data.DataLoader, device: str, num_classes: int,
                                  multi_class: bool = False, suppress_log: bool = False) -> float:

    acc_per_label, num_elems_per_label = [0] * num_classes, [0] * num_classes
    model.eval()
    for data, ground_truth_labels, _, _ in dl_test:
        data, ground_truth_labels = data.to(device), ground_truth_labels.to(device)
        output = model(data).squeeze()
        predicted_labels = None
        if multi_class:
            predicted_labels = output.argmax(dim=-1).squeeze()
        else:
            predicted_labels = (output > 0.5).int().squeeze()

        for i in range(num_classes):
            label_mask = ground_truth_labels == i
            acc_per_label[i] += (predicted_labels[label_mask] == ground_truth_labels[label_mask]).sum().item()
            num_elems_per_label[i] += label_mask.sum().item()

    acc_per_label = torch.tensor(acc_per_label) / torch.tensor(num_elems_per_label)
    macro_avg_acc = acc_per_label.mean().item()
    if not suppress_log:
        print("---- Macro averaged over labels accuracy ----")
        print(f"Macro average label accuracy = {macro_avg_acc:.4g}")

    return round(macro_avg_acc, 5)

def test_model_avg_and_wg_accuracy(model: torch.nn.Sequential, dl_test_grouped: torch.utils.data.DataLoader, device: str, num_groups: int,
                                   multi_class: bool = False, suppress_log: bool = False) -> tuple[float, float, int]:
    acc_per_group, num_elems_for_group = [0] * num_groups, [0] * num_groups
    model.eval()
    for data, ground_truth_labels, _, groups in dl_test_grouped:
        data, ground_truth_labels, groups = data.to(device), ground_truth_labels.to(device), groups.to(device)
        output = model(data).squeeze()
        predicted_labels = None
        if multi_class:
            predicted_labels = output.argmax(dim=-1).squeeze()
        else:
            predicted_labels = (output > 0.5).int().squeeze()

        for i in range(num_groups):
            group_mask = groups == i
            num_elems_for_group[i] += group_mask.sum().item()
            group_acc = (predicted_labels[group_mask] == ground_truth_labels[group_mask]).sum().item()
            acc_per_group[i] += group_acc

    acc_per_group = torch.tensor(acc_per_group) / torch.tensor(num_elems_for_group)
    macro_avg_group_acc = acc_per_group.mean().item()
    if not suppress_log:
        print("--- Model accuracy per group ---")
        print(f"Macro average group accuracy = {macro_avg_group_acc:.4g}")
        print(f"Min group accuracy = {acc_per_group.min().item():.4g}, group idx = {acc_per_group.argmin().item()}")

    return round(macro_avg_group_acc, 5), round(acc_per_group.min().item(), 5), acc_per_group.argmin().item()


def test_delta_input_robustness(dl_masked: torch.utils.data.DataLoader, model: torch.nn.Sequential, epsilon: float, delta: float,
    loss_fn: str, device: str, has_conv: bool = False, suppress_log: bool = False) -> tuple[float, float, float, float]:
    assert loss_fn in ["binary_cross_entropy", "cross_entropy"], "Only binary_cross_entropy and cross_entropy supported"
    # The model needs to be delta input robust only in the irrelevant features
    model.eval()
    num_robust, min_robust_delta, num_test_samples = 0, 0, 0
    max_upper_bound, min_lower_bound = 0, 0
    model = model.to(device).eval()
    for test_batch, test_labels, test_masks, _ in dl_masked:
        test_batch, test_labels, test_masks = test_batch.to(device), test_labels.to(device), test_masks.to(device)
        # The MLX method does not really matter, as we return the grads
        grad_bounds = input_gradient_interval_regularizer(model, test_batch, test_labels, loss_fn, epsilon, 0.0, return_grads=True,
            regularizer_type="r4", batch_masks=test_masks, has_conv=has_conv, device=device)
        d_l, d_u = grad_bounds[1]
        d_l, d_u = d_l * test_masks, d_u * test_masks
        for idx in range(len(test_batch)):
            num_salient_features = test_masks[idx].sum().item()
            # keep the dimensions, but remove the elements which are 0
            abs_diff = torch.abs(d_l[idx] - d_u[idx])
            min_robust_delta = max(min_robust_delta, abs_diff.max().item())
            max_upper_bound = max(max_upper_bound, d_u[idx].max().item())
            min_lower_bound = min(min_lower_bound, d_l[idx].min().item())
            robust_grad_inputs = torch.where(abs_diff <= delta, 1, 0)
            num_robust_salient_features = robust_grad_inputs.sum().item()
            num_robust += int(num_robust_salient_features == num_salient_features)
            num_test_samples += 1
    num_robust /= num_test_samples
    if not suppress_log:
        print("--- Mininimum delta for which the test set is certifiably 1-delta-input-robust ---")
        print(f"Min robust delta = {min_robust_delta:.3g}")
    # truncate to 3 decimal places
    num_robust, min_robust_delta = round(num_robust, 3), round(min_robust_delta, 3)
    return num_robust, min_robust_delta, min_lower_bound, max_upper_bound

def uniformize_magnitudes_schedule(curr_epoch: int, max_epochs: int, std_loss: float, rrr_loss: float) -> float:
    if curr_epoch < max_epochs // 5:
        return 0.0
    else:
        # get magnitude difference in terms of order of magnitude
        loss_diff = rrr_loss - std_loss
        if loss_diff < 0:
            return 1.0
        orders_of_mag = torch.floor(torch.log10(loss_diff))
        # the 2 is there to allow for a bit of a margin
        return 1 / (2 * (10 ** (orders_of_mag - 1)))

def write_results_to_file(filename: str, results: dict, method: str) -> None:
    assert filename.endswith(".yaml"), "Only yaml files supported"
    if not os.path.exists(filename):
        # Create a new file and write the dict to it
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump({method: results}, f)
    else:
        # Load the existing yaml file, append to it and write it back
        new_results = None
        with open(filename, "r", encoding="utf-8") as f:
            new_results = yaml.load(f, Loader=yaml.Loader) or {}
            new_results[method] = results
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(new_results, f)

def load_params_or_results_from_file(filename: str, method: str) -> dict:
    assert filename.endswith(".yaml"), "Only yaml files supported"
    assert os.path.exists(filename), "File does not exist"
    results = None
    with open(filename, "r", encoding="utf-8") as f:
        results = yaml.load(f, Loader=yaml.Loader)

    return results[method] if results is not None else None

def accumulate_model_with_certified_input_grad(
    dl_train: torch.utils.data.DataLoader,
    n_epochs: int,
    model: torch.nn.Module,
    learning_rate: float,
    criterion: torch.nn.Module,
    epsilon: float,
    mlx_method: str, # one of ["std", "grad_cert", "ibp_ex", "r3" or "r4"]
    k: float, # input reg weight
    device: str,
    has_conv: bool,
    k_schedule: callable = None,
    weight_reg_coeff: float = 0.0,
    suppress_tqdm: bool = False,
    num_accs = 1
) -> None:
    loss_fn = None
    if isinstance(criterion, torch.nn.BCELoss):
        loss_fn = "binary_cross_entropy"
    elif isinstance(criterion, torch.nn.CrossEntropyLoss):
        loss_fn = "cross_entropy"
    else:
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # pre-train the model
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", ) if not suppress_tqdm else range(n_epochs)
    model = model.to(device)
    loss = None
    for curr_epoch in progress_bar:
        for i, (x, u, m) in enumerate(dl_train):
            if (i > 0 and i % num_accs == 0) or i == len(dl_train) - 1:
                # Backward and optimize
                optimizer.zero_grad()
                loss = loss / num_accs
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    if not suppress_tqdm:
                        progress_bar.set_postfix({"loss": loss.item(), "reg": inp_grad_reg})
                    else:
                        print(f"Epoch {curr_epoch}, loss: {loss.item()}, reg: {inp_grad_reg}")
                loss = None
            # Forward pass
            u, x, m = u.to(device), x.to(device), m.to(device)

            # For std, we will waste some time doing the bounds, but at least it is consistent across methods
            inp_grad_reg = input_gradient_interval_regularizer(
                model, x, u, loss_fn, epsilon, 0.0, regularizer_type=mlx_method, batch_masks=m, has_conv=has_conv,
                device=device, weight_reg_coeff=weight_reg_coeff
            )
            if mlx_method == "std":
                assert inp_grad_reg == 0
                if loss_fn == "cross_entropy":
                # The last module is either Softmax or Sigmoid, hence why the [-2] is used
                    u = torch.nn.functional.one_hot(u, num_classes=list(model.modules())[-2].out_features).float()
            # output is [batch_size, 1], u is [bach_size] but BCELoss does not support different target and source sizes
            output = model(x).squeeze()
            std_loss = criterion(output, u)
            if k_schedule is not None:
                k = k_schedule(curr_epoch, n_epochs, std_loss.item(), inp_grad_reg)
            loss = std_loss + k * inp_grad_reg
            if i % 100 == 0:
                if not suppress_tqdm:
                    progress_bar.set_postfix({"loss": loss.item(), "reg": inp_grad_reg})
                else:
                    print(f"Epoch {curr_epoch}, loss: {loss.item()}, reg: {inp_grad_reg}")
