"""
This file implements the robust regularisation term from the paper 'Robust Explanation Constraints for Neural Networks'.
Nothing about this is 'certified', but we can use it for robust (but un-certified) pre-training.
"""
import torch
from torch.func import grad, vmap
from math import ceil, floor

import abstract_gradient_training.certified_training_utils as ct_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.bounds.loss_gradients import \
    bound_loss_function_derivative
from abstract_gradient_training.nominal_pass import (nominal_backward_pass,
                                                     nominal_forward_pass)
from .R4_models import SalientImageNet
from .llm import LLM, Tokenizer

#TODO it seems that for r3, ibp_ex and pgd_ex the smoothing does all the job? Is there something wrong in my code?
def input_gradient_interval_regularizer(
    model: torch.nn.Sequential,
    batch: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: str,
    epsilon: float,
    model_epsilon: float,
    return_grads: bool = False,
    regularizer_type: str = "std",
    batch_masks: torch.Tensor = None,
    has_conv: bool = False,
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0,
) -> torch.Tensor | list[tuple[torch.Tensor]]:
    """
    Compute an interval over the gradients of the loss with respect to the inputs to the network. Then compute the norm
    of this input gradient interval and return it to be used as a regularization term. This can be used for robust
    (but un-certified) pre-training.

    Args:
        model (torch.nn.Sequential): The neural network model.
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize].
        loss_fn (str): Only "binary_cross_entropy" supported for now.
        epsilon (float): The training time input perturbation budget.
        model_epsilon (float): The training time model perturbation budget.
        return_grads (bool): Whether to return the gradients directly, instead of the regularization term.
        regularizer_type (str): The type of regularizer to use. Can be "grad_cert", "r4", "r3", "std" or "ibp_ex".
        batch_masks (torch.Tensor): The salient image masks for the input data batch, should match the shape of the input.
        has_conv (bool): Whether the model has convolutional layers.
        device (str): The device to use for computation.

    Returns:
        float | list[torch.Tensor]: The regularization term or the gradient lower bounds.
    """
    # we only support binary cross entropy loss for now
    assert loss_fn in ["binary_cross_entropy", "cross_entropy"], f"Unsupported loss function: {loss_fn}"
    assert regularizer_type in ["grad_cert", "r4", "r3", "std", "ibp_ex", "ibp_ex+r3"], f"Unsupported regularizer type: {regularizer_type}"
    assert batch_masks is not None
    #! =========================== Checkpoint to make training faster ===========================
    if regularizer_type == "std":
        return 0
    #! ==================================== End of Checkpoint ===================================
    modules = list(model.modules())
    assert isinstance(modules[-1], torch.nn.Sigmoid) or isinstance(modules[-1], torch.nn.Softmax)
    last_layer_act_func = modules[-1]
    criterion = torch.nn.BCELoss() if loss_fn == "binary_cross_entropy" else torch.nn.CrossEntropyLoss()
    # remove the first module (copy of model) and the last module (sigmoid or softmax)
    # if the first module is a DataParallel, remove the first two instead
    modules = modules[2:-1] if isinstance(modules[0], torch.nn.DataParallel) else modules[1:-1]
    params_n_dense = ct_utils.get_parameters(model)[0]
    # do a forward pass without gradients to be able to call the loss gradient bounding functions in the agt module
    with torch.no_grad():
        logits_n = batch
        for module in modules:
            logits_n = module(logits_n)
        logits_n = logits_n.unsqueeze(-1)

    # ================================= BOUNDS COMPUTATION =================================
    intermediate = None
    if regularizer_type in ["r4", "ibp_ex", "ibp_ex+r3"]:
        intermediate = [(batch - epsilon * batch_masks, batch + epsilon * batch_masks)]
    else:
        intermediate = [(batch - epsilon, batch + epsilon)]
    # This is needed so that we can use the same uniform interface of the bounds to get the input gradient
    intermediate_nominal = [(batch, batch)]
    # propagate through the forward pass
    for module in modules:
        intermediate.append(propagate_module_forward(module, *intermediate[-1], model_epsilon))
        intermediate_nominal.append(propagate_module_forward(module, *intermediate_nominal[-1], 0))
        interval_arithmetic.validate_interval(*intermediate[-1], msg=f"forward pass {module}")

    logits_l, logits_u = intermediate[-1]
    #! =========================== Checkpoint to make training faster ===========================
    weight_sum = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
    for module in modules:
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    l2_reg = weight_reg_coeff * weight_sum
    if regularizer_type == "ibp_ex":
        y_bar = None
        if loss_fn == "binary_cross_entropy":
            y_bar = last_layer_act_func(logits_l.squeeze()) * labels + last_layer_act_func(logits_u.squeeze()) * (1 - labels)
        else: # cross entropy
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=modules[-1].out_features)
            # squeeze because logits are [batchsize x output_dim x 1]
            y_bar = last_layer_act_func(logits_l).squeeze() * labels_one_hot + last_layer_act_func(logits_u).squeeze() * (1 - labels_one_hot)
        return l2_reg + criterion(y_bar.squeeze(), labels)
    #! ==================================== End of Checkpoint ===================================
    # propagate through the loss function
    dl_l, dl_u, dl_n = bound_loss_function_derivative(loss_fn, logits_l, logits_u, logits_n, labels)
    dl_l, dl_u = dl_l.squeeze(-1), dl_u.squeeze(-1)
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="loss gradient")

    # propagate through the backward pass
    grads = [(dl_l, dl_u)]
    for i in range(len(modules) - 1, -1, -1):
        dl_l, dl_u = propagate_module_backward(modules[i], dl_l, dl_u, *intermediate[i], model_epsilon)
        interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")
        # This will happen when we have a Flatten() module at the beginning of the model
        if i == 0 and dl_l.shape != batch_masks.shape:
            dl_l, dl_u = dl_l.reshape(batch_masks.shape), dl_u.reshape(batch_masks.shape)
        grads.append((dl_l, dl_u))


    # ================================= INPUT GRAD COMPUTATION =================================
    # Need to unsqueeze the batch dimension if not conv to satisfy the quirks of this function (experimented on isic and decoy_mnist)
    input_grad = None
    if return_grads or regularizer_type in ["r3", "ibp_ex+r3"]:
        if has_conv:
            if loss_fn == "cross_entropy":
                dl_n = dl_n.squeeze(-1)
            for i in range(len(modules) - 1, -1, -1):
                dl_n, dl_n_1 = propagate_module_backward(modules[i], dl_n, dl_n, *intermediate_nominal[i], 0) # 0 model epsilon
                interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"backward pass {modules[i]}")
                assert torch.allclose(dl_n, dl_n_1)
            input_grad = dl_n
        else:
            activations_n = nominal_forward_pass(batch.flatten(start_dim=1).unsqueeze(-1), params_n_dense)
            input_grad = nominal_backward_pass(dl_n, params_n_dense, activations_n, with_input_grad=True)[0]
        # This will happen when we have a Flatten() module at the beginning of the model
        if input_grad.shape != batch_masks.shape:
            input_grad = input_grad.reshape(batch_masks.shape)
        grads.append((input_grad, None))

    if return_grads:
        grads.reverse()
        return grads

    # pre-compute the regularization term for code clarity
    match regularizer_type:
        case "grad_cert":
            # Loss for the interval regularizer
            return torch.norm(dl_u - dl_l, p=2) / dl_l.nelement()
        case "r4":
            # Loss for R4
            return torch.sum(torch.abs(torch.mul(dl_l, batch_masks)) +
            torch.abs(torch.mul(dl_u, batch_masks))) / dl_l.nelement()
        case "r3":
            # return the gradient of the loss w.r.t. the input squared and summed
            # input_grad_logits should now be of shape [batchsize x input_dim]
            reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
            # make sure the input_grad is of the same shape as the input
            reg_term = reg_term + torch.sum((input_grad.squeeze() * batch_masks) ** 2) / input_grad.nelement()
            return reg_term + l2_reg
        case "ibp_ex+r3":
            # ibp_ex term
            y_bar = None
            if loss_fn == "binary_cross_entropy":
                y_bar = last_layer_act_func(logits_l.squeeze()) * labels + last_layer_act_func(logits_u.squeeze()) * (1 - labels)
            else: # cross entropy
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=modules[-1].out_features)
                # squeeze because logits are [batchsize x output_dim x 1]
                y_bar = last_layer_act_func(logits_l).squeeze() * labels_one_hot + last_layer_act_func(logits_u).squeeze() * (1 - labels_one_hot)
            # r3 term
            reg_term = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
            reg_term = reg_term + torch.sum((input_grad.squeeze() * batch_masks) ** 2) / input_grad.nelement()
            # combine
            return l2_reg + criterion(y_bar.squeeze(), labels) + reg_term

def smooth_gradient_regularizer(
    batch: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    batch_masks: torch.Tensor,
    criterion: torch.nn.Module,
    epsilon: float,
    num_samples: int = 10,
    regularizer_type: str = "smooth_r3",
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0
) -> torch.Tensor:
    assert regularizer_type in ["smooth_r3", "rand_r4"]
    assert isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, torch.nn.BCELoss)

    batch = batch.unsqueeze(0).repeat(num_samples, *([1] * len(batch.shape)))
    sampling_dist = torch.distributions.uniform.Uniform(0, epsilon)
    perturbation = sampling_dist.sample(batch.shape).to(device)
    if regularizer_type == "rand_r4":
        perturbation *= batch_masks.unsqueeze(0) # should broadcast correctly
    perturbed_batch = batch + perturbation
    # Put the originat batch at the end of the perturbed batch so we can easily get its gradient
    perturbed_batch = torch.cat([perturbed_batch, batch[0].unsqueeze(0)], dim=0)

    def __compute_loss(sample: torch.Tensor) -> torch.Tensor:
        sample.requires_grad = True
        model.zero_grad()
        y_hat = model(sample)
        return criterion(y_hat.squeeze(), labels)

    grads = torch.tensor([]).to(device)
    if isinstance(model, SalientImageNet):
        # vectorization not supported for this pre-trained model
        for i in range(num_samples + 1):
            sample = perturbed_batch[i]
            sample.requires_grad = True
            model.zero_grad()
            y_hat = model(sample)
            loss_for_sample = criterion(y_hat.squeeze(), labels)
            loss_for_sample.backward()
            grads = torch.cat([grads, sample.grad.data.unsqueeze(0)], dim=0)
    else:
        grads = torch.vmap(grad(__compute_loss), in_dims=0)(perturbed_batch).to(device)
    perturbed_input_grad = grads[:-1]
    standard_input_grad = grads[-1]
    diff_input_grad = torch.abs(perturbed_input_grad - standard_input_grad)

    weight_sum = torch.tensor(0).to(device, dtype=torch.float32).requires_grad_()
    non_zero_masks = (batch_masks > 0).sum().item()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    match regularizer_type:
        case "smooth_r3":
            elem_wise_mean = torch.mean(diff_input_grad * batch_masks.unsqueeze(0), dim=0)
            return torch.norm(elem_wise_mean, p=2) / non_zero_masks + weight_reg_coeff * weight_sum
        case "rand_r4":
            elem_wise_max = torch.max(diff_input_grad * batch_masks.unsqueeze(0), dim=0).values
            return torch.norm(elem_wise_max, p=2) / non_zero_masks + weight_reg_coeff * weight_sum

def input_gradient_pgd_regularizer(
    batch: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    batch_masks: torch.Tensor,
    criterion: torch.nn.Module,
    epsilon: float,
    num_iterations: int = 10,
    regularizer_type: str = "std",
    device: str = "cuda:0",
    weight_reg_coeff: float = 0.0,
    clip_grad_bound: float = None,
) -> torch.Tensor:
    """This function is used to compute the adversarial perturbation budget for the input data batch and return it to be used as a regularization term
    in order to optimize the behaviour of a model to ignore irrelevant features. This is not a certification technique, but does provide a minimal level
    of robustness to the model to perturbations of the unsalient features.

    Args:
        batch (torch.Tensor): The input data batch, should have shape [batchsize x ... ].
        labels (torch.Tensor): The labels for the input data batch, should have shape [batchsize x output_dim] if cross entropy,
            [batchsize] if binary cross entropy.
        model (torch.nn.Module): The neural network model.
        batch_masks (torch.Tensor): The salient image masks for the input data batch, should match the shape of the input.
        loss_fn (str): The loss function to use. Supported: "binary_cross_entropy", "cross_entropy".
        epsilon (float): The adversarial perturbation budget.
        num_iterations (int, optional): Number of PGD iterations. Defaults to 10. If the number of iterations is 1, the attack is FGSM
        regularizer_type (str, optional): The type of regularizer to use. Can be "grad_cert", "r4", "r3", "std" or "pgd". Defaults to "std".
        device (_type_, optional): The device to use for computation. Defaults to "cuda:0".

    """
    assert regularizer_type in ["pgd_r4", "std", "r3"]
    assert batch_masks is not None
    #! =========================== Checkpoint to make training faster ===========================
    if regularizer_type == "std":
        return 0
    weight_sum = torch.tensor(0).to(dtype=torch.float32)
    weight_sum.requires_grad_()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            weight_sum = weight_sum + torch.sum(module.weight ** 2)
    if regularizer_type == "r3":
        model.zero_grad()
        reg_term = torch.tensor(0).to(dtype=torch.float32).requires_grad_()
        batch = batch.requires_grad_()
        y_hat = model(batch)
        loss = criterion(y_hat.squeeze(), labels)
        loss.backward()
        reg_term = reg_term + torch.sum((batch.grad.data.reshape(batch_masks.shape) * batch_masks) ** 2)
        return weight_reg_coeff * weight_sum + reg_term
    #! ==================================== End of Checkpoint ===================================

    # Only pgd_r4 from this point on
    pgd_adv_input = batch
    perturbation_masks = batch_masks
    for _ in range(num_iterations):
        pgd_adv_input.requires_grad = True
        # We need it because otherwise the gradients will be accumulated with previous iterations
        #@ This also means we need to perform the regularization BEFORE the normal training step
        model.zero_grad()

        # We need to squeeze the logits when the loss is BCELoss because labels has size [batchsize] and y_hat [batchsize x 1]
        loss = input_gradient_pgd_regularizer(
            pgd_adv_input, labels, model, batch_masks, criterion, epsilon, regularizer_type="r3"
        )
        loss.backward()

        adv_batch_step = pgd_adv_input + epsilon * perturbation_masks * torch.sign(pgd_adv_input.grad.data)
        delta = torch.clamp(adv_batch_step - batch, min=-epsilon, max=epsilon)
        pgd_adv_input = torch.clamp(batch + delta, min=batch.min(), max=batch.max()).detach_()

    # One last time, we do a full forward and backward pass to get the input gradient for the pgd adversarial example
    pgd_adv_input.requires_grad = True
    model.zero_grad()
    y_hat = model(pgd_adv_input)
    loss = criterion(y_hat.squeeze(), labels)
    # We squeeze here for the same reason as above
    loss.backward()
    pgd_grad_reg = torch.sum(torch.abs(torch.mul(pgd_adv_input.grad.data, batch_masks)))
    if clip_grad_bound is not None:
        pgd_grad_reg = torch.clamp(pgd_grad_reg, min=0, max=clip_grad_bound)

    return pgd_grad_reg + weight_reg_coeff * weight_sum

def _masked_token_indices(tokenizer: Tokenizer, batch_masks: list[str], token_ids: torch.Tensor) -> list[torch.Tensor]:
    batch_spur_word_indices = []
    for batch_elem_idx, mask in enumerate(batch_masks):
        if mask != "":
            spur_words = mask.split(",")
            curr_batch_idx_spur_indices = []
            for word in spur_words:
                word_token_ids = tokenizer.tokenize(word)["input_ids"]
                # Tokenizer might split a spurious word into multiple tokens if not in vocab
                for word_token_id in list(word_token_ids.flatten())[1:-1]: # No [CLS] and [SEP] tokens
                    #% We only select grad from the tokens associated with the current batch element considered
                    spur_word_indices = torch.where(token_ids[batch_elem_idx] == word_token_id)[0]
                    curr_batch_idx_spur_indices.append(spur_word_indices)
            # We want a list of tensors of spurious word indices, with 1 tensor for each text in the batch
            batch_spur_word_indices.append(torch.cat(curr_batch_idx_spur_indices, dim=0))
        else:
            # If no spurious words are provided, we add an empty tensor
            batch_spur_word_indices.append(torch.tensor([], dtype=torch.int64))

    return batch_spur_word_indices

def _replace_token_indices(
    tokenizer: Tokenizer,
    alpha: float,
    token_ids: torch.Tensor,
    batch_spur_word_indices: list[torch.Tensor] = None,
    device: str = "cuda:0"
) -> list[torch.Tensor]:
    new_token_ids = token_ids.clone().detach_()
    #% Smooth-R3 case:
    if batch_spur_word_indices is None:
        for row_idx, row in enumerate(token_ids):
            num_nonzero_tokens = (row != 0).sum()
            num_tokens_to_replace = int(num_nonzero_tokens * alpha)
            # Uniformly sample from the non-zero tokens excluding [CLS] and [SEP]
            weights = torch.ones_like(row[:num_nonzero_tokens], dtype=torch.float32) / num_nonzero_tokens
            row_sampled_indices = torch.multinomial(weights, num_tokens_to_replace, replacement=False)
            # Remove the indices where token_ids[index] == 101 [CLS], 102 [SEP]
            row_sampled_indices = row_sampled_indices[row[row_sampled_indices] != 101]
            row_sampled_indices = row_sampled_indices[row[row_sampled_indices] != 102]
            assert row_sampled_indices.ndim == 1
            # Randomly sample from vocab
            sampled_token_ids = tokenizer.sample_token_ids_from_vocab(row_sampled_indices.shape[0])
            new_token_ids[row_idx][row_sampled_indices] = sampled_token_ids.to(device)
    #% Rand-R4 case:
    else:
        for batch_elem_idx in range(len(batch_spur_word_indices)):
            num_spur_words = len(batch_spur_word_indices[batch_elem_idx])
            num_spur_words_to_replace = int(num_spur_words * alpha)
            if num_spur_words_to_replace == 0:
                continue
            # Uniformly sample from the spurious words token ids
            weights = torch.ones_like(batch_spur_word_indices[batch_elem_idx], dtype=torch.float32) / num_spur_words
            row_sampled_indices = torch.multinomial(weights, num_spur_words_to_replace, replacement=False)
            # Randomly sample from the set of spurious word token ids
            sampled_token_ids = tokenizer.sample_token_ids_from_spur_words(num_spur_words_to_replace)
            spur_selected_indices = batch_spur_word_indices[batch_elem_idx][row_sampled_indices]
            new_token_ids[batch_elem_idx][spur_selected_indices] = sampled_token_ids.to(device)

    return new_token_ids

def _get_chunked_adversary_set_gcg(tokenizer: Tokenizer, token_ids: torch.Tensor, alpha: float) -> tuple[torch.Tensor, list[torch.Tensor]]:
    superset_adversary_set_token_ids = tokenizer.spurious_words_token_ids
    adversary_set_size = int(floor(len(superset_adversary_set_token_ids) * alpha))
    adversary_set_token_ids = superset_adversary_set_token_ids[torch.randperm(len(superset_adversary_set_token_ids))[:adversary_set_size]]
    adv_set_batch_size = token_ids.shape[0]
    # chunk the adversary set token ids in batches of size batch_size to avoid OOM
    num_chunks = int(ceil(len(adversary_set_token_ids) / adv_set_batch_size))
    adversary_set_token_ids_chunks = list(adversary_set_token_ids.chunk(num_chunks))

    return adversary_set_token_ids, adversary_set_token_ids_chunks

@torch.enable_grad()
def _find_gcg_adversary(
    tokenizer: Tokenizer,
    model: LLM,
    criterion: torch.nn.Module,
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_spur_word_indices: list[torch.Tensor],
    alpha: float,
    device: str
) -> list[torch.Tensor]:
    embed_layer = model.embeddings
    adversarial_sentences = token_ids.clone().detach_()
    adversary_set_token_ids, adversary_set_token_ids_chunks = _get_chunked_adversary_set_gcg(tokenizer, token_ids, alpha)
    adv_criterion = torch.nn.BCELoss() if isinstance(criterion, torch.nn.BCELoss) else torch.nn.CrossEntropyLoss()
    for text_idx, spur_words_indices in enumerate(batch_spur_word_indices):
        for spur_word_idx in spur_words_indices:
            spur_word_grads = torch.tensor([]).to(device)
            for chunk_idx, adv_set_tids in enumerate(adversary_set_token_ids_chunks):
                # Don't want to deal with weird fragmentation and batching logic
                if len(adv_set_tids) <= 1:
                    continue
                # If token_ids is [batchsize x seq_len], then token_ids[text_idx] is [seq_len], so we can consider the len(adv_set_tids) as the batch size
                adversarial_sentence = token_ids[text_idx].unsqueeze(0).repeat(len(adv_set_tids), 1)
                adversarial_sentence[:, spur_word_idx] = adv_set_tids
                adv_tokens_embeds = embed_layer(adversarial_sentence).clone().detach()
                adv_tokens_embeds.requires_grad = True
                # Forward pass and get grads
                y_hat = model.inputs_embeds_forward(adv_tokens_embeds).squeeze()
                chunk_labels = torch.ones_like(y_hat, dtype=torch.float32).to(device) * labels[text_idx]
                # Since chunk_labels is a tensor of all ones/zeros, we don't need weights
                loss = adv_criterion(y_hat, chunk_labels)
                loss.backward()
                # Get gradients
                grads_token_embeds = adv_tokens_embeds.grad # [batch_size x seq_len x embedding_dim]
                # Collapse the embed dim to get a value proportional to the gradient for each token
                grads = torch.sum(grads_token_embeds, dim=-1) # [batch_size x seq_len]
                # Get the sentence with the highest total gradient for the spurious word index
                spur_word_grads_chunk = grads[:, spur_word_idx].squeeze().detach()
                del grads_token_embeds, grads, adversarial_sentence, y_hat, loss, adv_tokens_embeds
                assert spur_word_grads_chunk.ndim == 1
                spur_word_grads = torch.cat([spur_word_grads, spur_word_grads_chunk], dim=0)
            max_grad_idx = torch.argmax(spur_word_grads)
            # Replace word
            adversarial_sentences[text_idx][spur_word_idx] = adversary_set_token_ids[max_grad_idx]
            del spur_word_grads, max_grad_idx
        torch.cuda.empty_cache()

    return adversarial_sentences

@torch.enable_grad()
def llm_gradient_regularizer(
    model: LLM,
    tokenizer: Tokenizer,
    text: list[str],
    labels: torch.Tensor,
    batch_masks: list[str],
    criterion: torch.nn.Module,
    alpha: float = 0.1, # Percentage of random words to change for smooth-r3
    num_samples: int = 5, # Num samples for both smooth-r3 and rand-r4
    regularizer_type: str = "smooth_r3",
    device: str = "cuda:0",
) -> torch.Tensor:
    assert regularizer_type in ["std", "smooth_r3", "rand_r4", "r3", "pgd_r4"]
    assert isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, torch.nn.BCELoss)

    if regularizer_type == "std":
        return torch.tensor(0).to(device, dtype=torch.float32)

    encoding = tokenizer.tokenize(text)
    token_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    batch_spur_word_indices = _masked_token_indices(tokenizer, batch_masks, token_ids)
    perturbed_token_ids = token_ids.clone().detach_().unsqueeze(0).repeat(num_samples, *([1] * len(token_ids.shape)))
    match regularizer_type:
        case "smooth_r3":
            for i in range(num_samples):
                perturbed_token_ids[i] = _replace_token_indices(tokenizer, alpha, token_ids, device=device)
        case "rand_r4":
            for i in range(num_samples):
                perturbed_token_ids[i] = _replace_token_indices(tokenizer, alpha, token_ids, batch_spur_word_indices, device)
        case "r3":
            perturbed_token_ids = token_ids.unsqueeze(0)
        case "pgd_r4":
            perturbed_token_ids = _find_gcg_adversary(tokenizer, model, criterion, token_ids, labels, batch_spur_word_indices, alpha, device)
            perturbed_token_ids = perturbed_token_ids.unsqueeze(0)

    embed_layer = model.embeddings
    def __compute_loss(inputs_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds.requires_grad = True
        y_hat = model.inputs_embeds_forward(inputs_embeds)
        return criterion(y_hat.squeeze(), labels)

    perturbed_tokens_embeds = [embed_layer(perturbed_token_ids[i]).clone().detach() for i in range(perturbed_token_ids.shape[0])]
    perturbed_tokens_embeds = torch.stack(perturbed_tokens_embeds, dim=0).to(device)
    # Vectorize the loss computation and gradient
    perturbed_token_ids_grad = torch.vmap(grad(__compute_loss), in_dims=0, randomness="different")(perturbed_tokens_embeds).to(device)
    perturbed_token_ids_grad = torch.sum(perturbed_token_ids_grad, dim=-1) # get rid of the embedding dimension

    #% What I'm doing here is making a mask over token ids gradients, where 1 means spurious token id and 0 means non-spurious
    #% The mask has the exact same dimension as perturbed_token_ids_grad, so selecting spurious gradients just means multiplying
    spurious_grad_mask = torch.zeros_like(token_ids, dtype=torch.int8) # I.e. remove the sample dimension
    for batch_elem_idx in range(len(batch_masks)):
        spurious_grad_mask[batch_elem_idx][batch_spur_word_indices[batch_elem_idx]] = 1 # I.e. set the spurious words to 1
    spurious_grad_mask = spurious_grad_mask.unsqueeze(0).repeat(perturbed_token_ids_grad.shape[0], *([1] * len(spurious_grad_mask.shape)))
    masked_grads = perturbed_token_ids_grad * spurious_grad_mask

    match regularizer_type:
        case "smooth_r3":
            elem_wise_mean = torch.mean(masked_grads, dim=0)
            return torch.norm(elem_wise_mean, p=2) / masked_grads.nelement()
        case "rand_r4":
            elem_wise_max = torch.max(masked_grads, dim=0).values
            return torch.norm(elem_wise_max, p=2) / masked_grads.nelement()
        case "pgd_r4":
            return torch.norm(masked_grads, p=2) / masked_grads.nelement()
        case "r3":
            return torch.norm(masked_grads, p=2) / masked_grads.nelement()


def propagate_module_forward(
    module: torch.nn.Module, x_l: torch.Tensor, x_u: torch.Tensor, model_epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Propagate the input interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        x_l (torch.Tensor): Lower bound on the input.
        x_u (torch.Tensor): Upper bound on the input.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds on the output of the module.
    """
    if isinstance(module, torch.nn.Linear):
        W, b = module.weight, module.bias
        b = torch.zeros(W.size(0), 1, device=x_l.device, dtype=x_l.dtype) if b is None else b.unsqueeze(-1)
        W_l, W_u = W - model_epsilon, W + model_epsilon
        b_l, b_u = b - model_epsilon, b + model_epsilon
        if x_l.dim() == 2:
            x_l, x_u = x_l.unsqueeze(-1), x_u.unsqueeze(-1)
        x_l, x_u = interval_arithmetic.propagate_affine(x_l, x_u, W_l, W_u, b_l, b_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = torch.relu(x_l), torch.relu(x_u)
    elif isinstance(module, torch.nn.Conv2d):
        W, b = module.weight, module.bias
        W_l, W_u = W - model_epsilon, W + model_epsilon
        b_l, b_u = b - model_epsilon, b + model_epsilon
        dilation, stride, padding = module.dilation, module.stride, module.padding
        x_l, x_u = interval_arithmetic.propagate_conv2d(
            x_l, x_u, W_l, W_u, b_l, b_u, stride=stride, padding=padding, dilation=dilation
        )
    elif isinstance(module, torch.nn.Flatten):
        x_l, x_u = module(x_l), module(x_u)
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(x_l, x_u, msg=f"module {module}")
    return x_l, x_u


def propagate_module_backward(
    module: torch.nn.Module,
    dl_l: torch.Tensor,
    dl_u: torch.Tensor,
    x_l: torch.Tensor,
    x_u: torch.Tensor,
    model_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Back-propagate the gradient interval through the given nn module.

    Args:
        module (torch.nn.Module): The module to propagate through.
        dl_l (torch.Tensor): Lower bound on the gradient of the loss wrt the output of this module.
        dl_u (torch.Tensor): Upper bound on the  gradient of the loss wrt the output of this module.
        x_l (torch.Tensor): Lower bound on the input to the module.
        x_u (torch.Tensor): Upper bound on the input to the module.
        model_epsilon (float): The model perturbation budget.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Lower and upper bound on the gradient of the loss wrt the module input.
    """
    interval_arithmetic.validate_interval(dl_l, dl_u, msg="dl input")
    if isinstance(module, torch.nn.Linear):
        W = module.weight
        W_l, W_u = W - model_epsilon, W + model_epsilon
        dl_l, dl_u = interval_arithmetic.propagate_matmul(dl_l, dl_u, W_l, W_u)
    elif isinstance(module, torch.nn.ReLU):
        x_l, x_u = x_l.reshape(dl_l.size()), x_u.reshape(dl_u.size())
        dl_l, dl_u = interval_arithmetic.propagate_elementwise(dl_l, dl_u, (x_l > 0).float(), (x_u > 0).float())
    elif isinstance(module, torch.nn.Conv2d):
        W, dilation, stride = module.weight, module.dilation, module.stride
        padding, groups = module.padding, module.groups
        W_l, W_u = W - model_epsilon, W + model_epsilon

        # function that computes the gradient of the convolution wrt the input
        def conv_grad(dl_, W_):
            return torch.nn.grad.conv2d_input(x_l.shape, W_, dl_, stride, padding, dilation, groups)

        dl_l, dl_u = interval_arithmetic.propagate_linear_transform(dl_l, dl_u, W_l, W_u, transform=conv_grad)
    elif isinstance(module, torch.nn.Flatten):
        dl_l, dl_u = torch.reshape(dl_l, x_l.size()), torch.reshape(dl_u, x_u.size())
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
    interval_arithmetic.validate_interval(dl_l, dl_u, msg=f"module {module}")
    return dl_l, dl_u

# if __name__ == "__main__":
#     # test the gradient calculations vs torch autograd
#     import sys
#
#     sys.path.append("..")
#     from datasets.oct_mnist import \
#         get_dataloaders  # pylint: disable=import-error
#     from deepmind import DeepMindSmall
#
#     # define model, dataset and optimizer
#     device = "cuda:0"
#     torch.manual_seed(0)
#     test_model = DeepMindSmall(1, 1)
#     dataloader, _ = get_dataloaders(64)
#     criterion = torch.nn.BCELoss(reduction="sum")
#     test_model = test_model.to(device)
#
#     # get the input data and pass through the model and loss
#     test_batch, test_labels = next(iter(dataloader))
#     test_batch, test_labels = test_batch.to(device), test_labels.to(device)
#     test_batch.requires_grad = True
#     intermediates = [test_batch]
#     for layer in list(test_model.modules())[1:]:
#         intermediates.append(layer(intermediates[-1]))
#         intermediates[-1].retain_grad()
#     loss = criterion(intermediates[-1].squeeze().float(), test_labels.squeeze().float())
#     loss.backward()
#     autograds_intermediate = [inter.grad for inter in intermediates]
#     autograds_parameters = [param.grad for param in test_model.parameters()]
#
#     # ======================== test the input_gradient_interval_regularizer ========================
#
#     # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
#     custom_grads = input_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
#     )
#
#     for j in range(len(custom_grads)):
#         print(f"Layer {j}: matching gradients = {torch.allclose(custom_grads[j], autograds_intermediate[j])}")
#
#     # gradient regularization term:
#     reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
#     print("Input gradient interval regularization term (input perturbation):", reg)
#     reg = input_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001)
#     print("Input gradient interval regularization term (model perturbation):", reg)
#
#     # ======================== test the parameter_gradient_interval_regularizer ========================
#
#     # pass the data through our gradient interval regularizer with zero epsilon, which should match the exact grads
#     custom_grads = parameter_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.0, return_grads=True
#     )
#
#     for j in range(len(custom_grads)):
#         print(f"Parameter {j}: matching gradients = {torch.allclose(custom_grads[j], autograds_parameters[j])}")
#
#     # gradient regularization term:
#     reg = parameter_gradient_interval_regularizer(test_model, test_batch, test_labels, "binary_cross_entropy", 0.1, 0.0)
#     print("Parameter gradient interval regularization term (input perturbation):", reg)
#     reg = parameter_gradient_interval_regularizer(
#         test_model, test_batch, test_labels, "binary_cross_entropy", 0.0, 0.001
#     )
#     print("Parameter gradient interval regularization term (model perturbation):", reg)
#