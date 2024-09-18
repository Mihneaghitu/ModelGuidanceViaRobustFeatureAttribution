"""Certified privacy training."""

from __future__ import annotations
from typing import Optional, Callable
import logging

import torch
from torch.utils.data import DataLoader

from abstract_gradient_training import nominal_pass
from abstract_gradient_training import certified_training_utils as ct_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training import optimizers


LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def privacy_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: DataLoader,
    dl_test: DataLoader,
    transform: Optional[Callable] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Train the dense layers of a neural network with the given config and return the privacy-certified bounds
    on the parameters.

    NOTE: The returned nominal parameters are not guaranteed to be inside the parameter bounds if dp_sgd is used.

    Args:
        model (torch.nn.Sequential): Neural network model. Must be a torch.nn.Sequential object with dense layers and
                                     ReLU activations only. The model may have other layers (e.g. convolutional layers)
                                     before the dense section, but these must be fixed and are not trained. If fixed
                                     non-dense layers are provided, then the transform function must be set to propagate
                                     bounds through these layers.
        config (AGTConfig): Configuration object for the abstract gradient training module. See the configuration module
                            for more details.
        dl_train (DataLoader): Training data loader.
        dl_test (DataLoader): Testing data loader.
        transform (Optional[Callable], optional): Optional function to propagate bounds through fixed layers of the
                                                  neural network (e.g. convolutional layers). Defaults to None.

    Returns:
        param_l (list[torch.Tensor]): List of lower bounds of the trained parameters [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of nominal trained parameters [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of upper bounds of the trained parameters [W1, b1, ..., Wn, bn].
    """

    # initialise hyperparameters, model, data, optimizer, logging
    device = torch.device(config.device)
    model = model.to(device)  # match the device of the model and data
    param_n, param_l, param_u = ct_utils.get_parameters(model)
    optimizer = optimizers.SGD(config)
    gamma = config.clip_gamma
    sigma = config.dp_sgd_sigma
    k_private = config.k_private

    # set up logging
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info("=================== Starting Privacy Certified Training ===================")
    LOGGER.debug(
        "\tPrivacy parameters: k_private=%s, clip_gamma=%s, dp_sgd_sigma=%s",
        config.k_private,
        config.clip_gamma,
        config.dp_sgd_sigma,
    )
    LOGGER.debug("\tBounding methods: forward=%s, backward=%s", config.forward_bound, config.backward_bound)

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = ct_utils.dataloader_wrapper(dl_train, config.n_epochs)

    for n, (batch, labels) in enumerate(training_iterator):
        # evaluate the network and log the results
        network_eval = config.test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)
        # get if we should terminate training early
        if config.early_stopping and ct_utils.break_condition(network_eval):
            break
        LOGGER.info("Training batch %s: %s", n, ct_utils.get_progress_message(network_eval, param_l, param_u))
        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1).type(param_n[-1].dtype)
        batchsize = batch.size(0)
        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        grads_l = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        grads_l_top_ks = [[] for _ in param_n]  # top k lower bound gradients from each fragment
        grads_u_bottom_ks = [[] for _ in param_n]  # bottom k upper bound gradients from each fragment

        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for f in range(len(batch_fragments)):  # loop over all batch fragments
            batch_frag = batch_fragments[f].to(device)
            batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
            label_frag = label_fragments[f].to(device)
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            logit_n = activations_n[-1]
            _, _, dL_n = config.loss_bound_fn(logit_n, logit_n, logit_n, label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, activations_n)
            frag_grads_l, frag_grads_u = ct_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config
            )

            # accumulate the results for this batch to save memory
            for i in range(len(grads_n)):
                # clip the gradients
                frag_grads_l[i], frag_grads_n[i], frag_grads_u[i] = ct_utils.propagate_clipping(
                    frag_grads_l[i], frag_grads_n[i], frag_grads_u[i], gamma, config.clip_method
                )
                # accumulate the nominal gradients
                grads_n[i] += frag_grads_n[i].sum(dim=0)
                # accumulate the top/bottom s - k gradient bounds
                size = frag_grads_n[i].size(0)
                # we are guaranteed to take the bottom s - k from the lower bound, so add the sum to grads_l
                # the remaining k gradients are stored until all the frags have been processed
                top_k_l = torch.topk(frag_grads_l[i], min(size, k_private), largest=True, dim=0)[0]
                grads_l[i] += frag_grads_l[i].sum(dim=0) - top_k_l.sum(dim=0)
                grads_l_top_ks[i].append(top_k_l)

                # we are guaranteed to take the top s - k from the upper bound, so add the sum to grads_u
                # the remaining k gradients are stored until all the frags have been processed
                bottom_k_u = torch.topk(frag_grads_u[i], min(size, k_private), largest=False, dim=0)[0]
                grads_u[i] += frag_grads_u[i].sum(dim=0) - bottom_k_u.sum(dim=0)
                grads_u_bottom_ks[i].append(bottom_k_u)

        # concatenate
        grads_l_top_ks = [torch.cat(g, dim=0) for g in grads_l_top_ks]
        grads_u_bottom_ks = [torch.cat(g, dim=0) for g in grads_u_bottom_ks]

        # Apply the unlearning update mechanism to the bounds.
        for i in range(len(grads_n)):
            size = grads_l_top_ks[i].size(0)
            assert size >= k_private, "Not enough samples left after processing batch fragments."
            top_k_l = torch.topk(grads_l_top_ks[i], k_private, largest=True, dim=0)[0]
            bottom_k_u = torch.topk(grads_u_bottom_ks[i], k_private, largest=False, dim=0)[0]
            grads_l[i] += grads_l_top_ks[i].sum(dim=0) - top_k_l.sum(dim=0) - k_private * gamma
            grads_u[i] += grads_u_bottom_ks[i].sum(dim=0) - bottom_k_u.sum(dim=0) + k_private * gamma

        # normalise each by the batchsize
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        # check bounds and add noise
        for i in range(len(grads_n)):
            if sigma == 0.0:  # sound update
                interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])
            else:  # unsound update due to noise
                interval_arithmetic.validate_interval(grads_l[i], grads_u[i])
            grads_n[i] += torch.normal(torch.zeros_like(grads_n[i]), sigma)

        param_n, param_l, param_u = optimizer.step(param_n, param_l, param_u, grads_n, grads_l, grads_u, sound=False)

    network_eval = config.test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)
    LOGGER.info("Final network eval: %s", ct_utils.get_progress_message(network_eval, param_l, param_u))

    for i in range(len(param_n)):
        violations = (param_l[i] > param_n[i]).sum() + (param_n[i] > param_u[i]).sum()
        max_violation = max((param_l[i] - param_n[i]).max(), (param_n[i] - param_u[i]).max())
        if violations > 0:
            LOGGER.info("Nominal parameters not within certified bounds for parameter %s due to DP-SGD noise.", i)
            LOGGER.debug("\tNumber of violations: %s", violations.item())
            LOGGER.debug("\tMax violation: %s", max_violation.item())

    LOGGER.info("=================== Finished Privacy Certified Training ===================")

    return param_l, param_n, param_u
