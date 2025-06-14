"""Poison certified training."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from typing import Union

import torch
from torch.utils.data import DataLoader

from abstract_gradient_training import certified_training_utils as ct_utils
from abstract_gradient_training import (interval_arithmetic, nominal_pass,
                                        optimizers)
from abstract_gradient_training.configuration import AGTConfig

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def poison_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: DataLoader,
    dl_test: DataLoader,
    dl_clean: DataLoader | None = None,
    transform: Callable | None = None,
    return_input_grad: bool = False
) -> Union[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]], # when we return grad_input
           tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]: # when we only want bounds + nominal
    """
    Train the neural network while tracking lower and upper bounds on all the parameters under a possible poisoning
    attack.

    Args:
        model (torch.nn.Sequential): Neural network model to train. Expected to be a Sequential model with linear
            layers and ReLU activations on all layers except the last. Models with (fixed) convolutional layers are
            also accepted but the transform function must be provided to handle the propagation through these layers.
        config (ct_config.AGTConfig): Configuration object (see agt.certified_training.configuration.py for details)
        dl_train (DataLoader): Dataloader for the training data.
        dl_test (DataLoader): Dataloader for the testing data.
        dl_clean (DataLoader, optional): Dataloader for "clean" training data. If provided, a batch will be taken from
            both dl_train and dl_clean for each training batch. Poisoned bounds will only be calculated for the batches
            from dl_train.
        transform (Callable): Function that transforms and bounds the input data for any initial, fixed, non-affine
            layers of the neural network. For example, propagating bounds through fixed
            convolutional layers. Defaults to None.

    Returns:
        param_l: list of lower bounds on the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
        param_n: list of nominal values of the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
        param_u: list of upper bounds on the parameters of the linear layers of the model [W1, b1, ..., Wm, bm]
    """

    # initialise hyperparameters, model, data, optimizer, logging
    device = torch.device(config.device)
    model = model.to(device)  # match the device of the model and data
    param_n, param_l, param_u = ct_utils.get_parameters(model)
    k_poison = max(config.k_poison, config.label_k_poison)
    optimizer = optimizers.SGD(config)

    # set up logging and print run info
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info("=================== Starting Poison Certified Training ===================")
    LOGGER.debug(
        "\tOptimizer params: n_epochs=%s, learning_rate=%s, l1_reg=%s, l2_reg=%s",
        config.n_epochs,
        config.learning_rate,
        config.l1_reg,
        config.l2_reg,
    )
    LOGGER.debug(
        "\tLearning rate schedule: lr_decay=%s, lr_min=%s, early_stopping=%s",
        config.lr_decay,
        config.lr_min,
        config.early_stopping,
    )
    LOGGER.debug("\tAdversary feature-space budget: epsilon=%s, k_poison=%s", config.epsilon, config.k_poison)
    LOGGER.debug(
        "\tAdversary label-space budget: label_epsilon=%s, label_k_poison=%s, poison_target=%s",
        config.label_epsilon,
        config.label_k_poison,
        config.poison_target,
    )
    LOGGER.debug("\tClipping: gamma=%s, method=%s", config.clip_gamma, config.clip_method)
    LOGGER.debug(
        "\tBounding methods: forward=%s, loss=%s, backward=%s", config.forward_bound, config.loss, config.backward_bound
    )

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = ct_utils.dataloader_pair_wrapper(dl_train, dl_clean, config.n_epochs, param_n[-1].dtype)
    test_iterator = itertools.cycle(dl_test)

    for n, (batch, labels, batch_clean, labels_clean) in enumerate(training_iterator):
        # evaluate the network
        network_eval = config.test_loss_fn(
            param_n, param_l, param_u, *next(test_iterator), model=model, transform=transform
        )

        # possibly terminate early
        if config.early_stopping and ct_utils.break_condition(network_eval):
            break
        config.callback(network_eval, param_l, param_n, param_u)

        # log the current network evaluation
        LOGGER.info("Training batch %s: %s", n + 1, ct_utils.get_progress_message(network_eval, param_l, param_u))

        # calculate batchsize
        batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)

        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1)
            batch_clean = batch_clean.view(batch_clean.size(0), -1, 1) if batch_clean is not None else None

        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradients
        grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradients
        grads_diffs_l = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds
        grads_diffs_u = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds

        # process clean data
        batch_fragments = torch.split(batch_clean, config.fragsize, dim=0) if batch_clean is not None else []
        label_fragments = torch.split(labels_clean, config.fragsize, dim=0) if labels_clean is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
            # nominal pass
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            _, _, dL_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, activations_n)
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = ct_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config, False
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i] = ct_utils.propagate_clipping(
                    frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the gradients
                grads_l[i] = grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
                grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                grads_u[i] = grads_u[i] + frag_grads_wp_u[i].sum(dim=0)

        # process potentially poisoned data
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            # nominal pass
            batch_frag_n = transform(batch_frag, model, 0)[0] if transform else batch_frag
            activations_n = nominal_pass.nominal_forward_pass(batch_frag_n, param_n)
            _, _, dL_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, activations_n)
            # weight perturbed bounds
            frag_grads_wp_l, frag_grads_wp_u = ct_utils.grads_helper(
                batch_frag_n, batch_frag_n, label_frag, param_l, param_u, config, False
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i] = ct_utils.propagate_clipping(
                    frag_grads_wp_l[i], frag_grads_n[i], frag_grads_wp_u[i], config.clip_gamma, config.clip_method
                )
                # accumulate the gradients
                grads_l[i] = grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
                grads_n[i] = grads_n[i] + frag_grads_n[i].sum(dim=0)
                grads_u[i] = grads_u[i] + frag_grads_wp_u[i].sum(dim=0)

            # apply input transformation
            if transform:
                batch_frag_l, batch_frag_u = transform(batch_frag, model, config.epsilon)
            else:
                batch_frag_l, batch_frag_u = batch_frag - config.epsilon, batch_frag + config.epsilon
            # input + weight perturbed bounds
            frag_grads_iwp_l, frag_grads_iwp_u = ct_utils.grads_helper(
                batch_frag_l, batch_frag_u, label_frag, param_l, param_u, config, True
            )
            # clip and accumulate the gradients
            for i in range(len(grads_n)):
                frag_grads_iwp_l[i], _, frag_grads_iwp_u[i] = ct_utils.propagate_clipping(
                    frag_grads_iwp_l[i], torch.zeros(1), frag_grads_iwp_u[i], config.clip_gamma, config.clip_method
                )
                # calculate the differences beetween the input+weight perturbed and weight perturbed bounds
                diffs_l = frag_grads_iwp_l[i] - frag_grads_wp_l[i]
                diffs_u = frag_grads_iwp_u[i] - frag_grads_wp_u[i]
                # accumulate and store the the top-k diffs from each fragment
                grads_diffs_l[i].append(torch.topk(diffs_l, k_poison, dim=0, largest=False)[0])
                grads_diffs_u[i].append(torch.topk(diffs_u, k_poison, dim=0)[0])

        # accumulate the top-k diffs from each fragment then add the overall top-k diffs to the gradient bounds
        for i in range(len(grads_n)):
            # we pop, process and del each one by one to save memory
            grads_diffs_l_i = grads_diffs_l.pop(0)
            grads_diffs_l_i = torch.cat(grads_diffs_l_i, dim=0)
            grads_l[i] += torch.topk(grads_diffs_l_i, k_poison, dim=0, largest=False)[0].sum(dim=0)
            del grads_diffs_l_i
            grads_diffs_u_i = grads_diffs_u.pop(0)
            grads_diffs_u_i = torch.cat(grads_diffs_u_i, dim=0)
            grads_u[i] += torch.topk(grads_diffs_u_i, k_poison, dim=0)[0].sum(dim=0)
            del grads_diffs_u_i
            interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])

        # normalise each by the batchsize
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        param_n, param_l, param_u = optimizer.step(param_n, param_l, param_u, grads_n, grads_l, grads_u)

    network_eval = config.test_loss_fn(
        param_n, param_l, param_u, *next(test_iterator), model=model, transform=transform
    )
    if return_input_grad:
        input_grads_per_fragment = []
        ordered_inputs = []
        ordered_labels = []
        # do one last pass and just get the input gradients

        for batch_frag, label_frag in dl_train:
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            input_shape  = param_n[0].shape[1:]
            batch_size = batch_frag.shape[0]
            # reshape so it doesn't fail asserts
            batch_frag = batch_frag.view(batch_size, *input_shape, 1)
            # nominal pass
            batch_frag_n = transform(batch_frag, model, 0)[0] if transform else batch_frag
            activations_n = nominal_pass.nominal_forward_pass(batch_frag_n, param_n)
            _, _, dl_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            logits_grad = config.last_layer_activation_grad(activations_n[-1])
            frag_input_grad = nominal_pass.nominal_input_gradient(param_n, activations_n, logits_grad)
            frag_grads_n = nominal_pass.nominal_backward_pass(dl_n, param_n, activations_n)

            input_grads_per_fragment.append(frag_input_grad)
            ordered_inputs.append(batch_frag)
            ordered_labels.append(label_frag)


        return param_l, param_n, param_u, input_grads_per_fragment, ordered_inputs, ordered_labels
    LOGGER.info("Final network eval: %s", ct_utils.get_progress_message(network_eval, param_l, param_u))

    LOGGER.info("=================== Finished Poison Certified Training ===================")

    return param_l, param_n, param_u
