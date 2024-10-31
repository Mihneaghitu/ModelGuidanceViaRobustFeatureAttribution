import torch
import tqdm
import sys
from .robust_regularizer import input_gradient_interval_regularizer, input_gradient_pgd_regularizer
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
    suppress_tqdm: bool = False
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
            std_loss = criterion(output, u)
            if k_schedule is not None:
                k = k_schedule(curr_epoch, n_epochs, std_loss.item(), inp_grad_reg)
            loss = std_loss + k * inp_grad_reg
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                if not suppress_tqdm:
                    progress_bar.set_postfix({"loss": loss.item(), "reg": inp_grad_reg})
                else:
                    print(f"Epoch {curr_epoch}, loss: {loss.item()}, reg: {inp_grad_reg}")


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
    weight_reg_coeff: float = 0.0
) -> None:
    if not (isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.CrossEntropyLoss)):
        raise ValueError("Criterion not supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # pre-train the model
    progress_bar = tqdm.trange(n_epochs, desc="Epoch", )
    model = model.to(device).train()
    for _ in progress_bar:
        for i, (x, u, m) in enumerate(dl_train):
            # Forward pass
            u, x, m = u.to(device), x.to(device), m.to(device)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                # The last module is either Softmax or Sigmoid, hence why the [-2] is used
                u = torch.nn.functional.one_hot(u, num_classes=list(model.modules())[-2].out_features).float()
            # For std, we will waste some time doing the bounds, but at least it is consistent across methods
            inp_grad_reg = input_gradient_pgd_regularizer(
                x, u, model, m, criterion, epsilon, num_iterations=10, regularizer_type=mlx_method, device=device, weight_reg_coeff=weight_reg_coeff
            )
            if mlx_method == "std":
                assert inp_grad_reg == 0
            # output is [batch_size, 1], u is [bach_size] but BCELoss does not support different target and source sizes
            output = model(x).squeeze()
            std_loss = criterion(output, u)
            loss = std_loss + k * inp_grad_reg
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                progress_bar.set_postfix({"loss": loss.item(), "reg": inp_grad_reg})

def test_model_accuracy(model: torch.nn.Sequential, dl_test: torch.utils.data.DataLoader, device: str,
                        multi_class: bool = False, suppress_log: bool = False) -> float:
    all_acc = 0
    num_inputs = 0
    for test_batch, test_labels, _ in dl_test:
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
        print(f"Nominal = {all_acc:.2g}")

    return round(all_acc, 3)

def test_delta_input_robustness(dl_masked: torch.utils.data.DataLoader, model: torch.nn.Sequential, epsilon: float, delta: float,
    loss_fn: str, device: str, has_conv: bool = False, suppress_log: bool = False) -> tuple[float, float, float, float]:
    assert loss_fn in ["binary_cross_entropy", "cross_entropy"], "Only binary_cross_entropy and cross_entropy supported"
    # The model needs to be delta input robust only in the irrelevant features
    num_robust, min_robust_delta, num_test_samples = 0, 0, 0
    max_upper_bound, min_lower_bound = 0, 0
    model = model.to(device).eval()
    for test_batch, test_labels, test_masks in dl_masked:
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
        print("--- Delta input robustness ---")
        print(f"Delta Input Robustness = {num_robust:.2g}")
        print("--- Mininimum delta for which the test set is certifiably 1-delta-input-robust ---")
        print(f"Min robust delta = {min_robust_delta:.3g}")
    # truncate to 3 decimal places
    num_robust, min_robust_delta = round(num_robust, 3), round(min_robust_delta, 3)
    return num_robust, min_robust_delta, min_lower_bound, max_upper_bound

def uniformize_magnitudes_schedule(curr_epoch: int, max_epochs: int, std_loss: float, rrr_loss: float) -> float:
    if curr_epoch < 0: # max_epochs // 5:
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
        with open(filename, "w") as f:
            yaml.dump({method: results}, f)
    else:
        # Load the existing yaml file, append to it and write it back
        new_results = None
        with open(filename, "r") as f:
            new_results = yaml.load(f, Loader=yaml.FullLoader) or {}
            new_results[method] = results
        with open(filename, "w") as f:
            yaml.dump(new_results, f)

def load_params_or_results_from_file(filename: str, method: str) -> dict:
    assert filename.endswith(".yaml"), "Only yaml files supported"
    assert os.path.exists(filename), "File does not exist"
    results = None
    with open(filename, "r") as f:
        results = yaml.load(f, Loader=yaml.FullLoader)

    return results[method] if results is not None else None
