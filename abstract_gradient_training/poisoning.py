"""Poison certified training."""

from __future__ import annotations
import logging
from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader

from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training import certified_training_utils as ct_utils
from abstract_gradient_training import nominal_pass
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training import optimizers

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def poison_certified_training(
    model: torch.nn.Sequential,
    config: AGTConfig,
    dl_train: DataLoader,
    dl_test: DataLoader,
    dl_clean: Optional[DataLoader] = None,
    transform: Optional[Callable] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Train the neural network while tracking lower and upper bounds on all the parameters under a possible poisoning
    attack.

    Args:
        model (torch.nn.Sequential): Neural network model to train. Expected to be a Sequential model with linear
                                     layers and ReLU activations on all layers except the last.
                                     Models with (fixed) convolutional layers are also accepted but the transform
                                     function must be provided to handle the propagation through these layers.
        config (ct_config.AGTConfig): Configuration object (see agt.certified_training.configuration.py for details)
        dl_train (torch.utils.data.DataLoader): Dataloader for the training data.
        dl_test (torch.utils.data.DataLoader): Dataloader for the testing data.
        dl_clean (torch.utils.data.DataLoader): Dataloader for "clean" training data. If provided, a batch will be
                                                taken from both dl_train and dl_clean for each training batch.
                                                Poisoned bounds will only be calculated for the batches from dl_train.
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

    # set up logging
    logging.getLogger("abstract_gradient_training").setLevel(config.log_level)
    LOGGER.info("=================== Starting Poison Certified Training ===================")
    LOGGER.debug(
        "\tAdversary budget: epsilon=%s, k_poison=%s, label_epsilon=%s, label_k_poison=%s",
        config.epsilon,
        config.k_poison,
        config.label_epsilon,
        config.label_k_poison,
    )
    LOGGER.debug("\tBounding methods: forward=%s, backward=%s", config.forward_bound, config.backward_bound)

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = ct_utils.dataloader_pair_wrapper(dl_train, dl_clean, config.n_epochs)

    for n, (batch, labels, batch_clean, labels_clean) in enumerate(training_iterator):
        # evaluate the network
        network_eval = config.test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)

        # possibly terminate early
        if config.early_stopping and ct_utils.break_condition(network_eval):
            break

        # log the current network evaluation
        LOGGER.info("Training batch %s: %s", n, ct_utils.get_progress_message(network_eval, param_l, param_u))

        # calculate batchsize
        batchsize = batch.size(0) if batch_clean is None else batch.size(0) + batch_clean.size(0)

        # we want the shape to be [batchsize x input_dim x 1]
        if transform is None:
            batch = batch.view(batch.size(0), -1, 1).type(param_n[-1].dtype)
            batch_clean = batch_clean.view(batch_clean.size(0), -1, 1).type(param_n[-1].dtype) if dl_clean else None

        # initialise containers to store the nominal and bounds on the gradients for each fragment
        # the bounds are stored as lists of lists indexed by [parameter, fragment]
        grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradients
        grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradients
        grads_diffs_l = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds
        grads_diffs_u = [[] for _ in param_n]  # difference of input+weight perturbed and weight perturbed bounds

        # process clean data
        batch_fragments = torch.split(batch_clean, config.fragsize, dim=0) if dl_clean else []
        label_fragments = torch.split(labels_clean, config.fragsize, dim=0) if dl_clean else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            batch_frag, label_frag = batch_frag.to(device), label_frag.to(device)
            batch_frag = transform(batch_frag, model, 0)[0] if transform else batch_frag
            # nominal pass
            activations_n = nominal_pass.nominal_forward_pass(batch_frag, param_n)
            _, _, dL_n = config.loss_bound_fn(activations_n[-1], activations_n[-1], activations_n[-1], label_frag)
            frag_grads_n = nominal_pass.nominal_backward_pass(dL_n, param_n, activations_n)
            # weight perturbed bounds
            grads_weight_perturb_l, grads_weight_perturb_u = ct_utils.grads_helper(
                batch_frag, batch_frag, label_frag, param_l, param_u, config, False
            )
            grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]
            grads_l = [a + b.sum(dim=0) for a, b in zip(grads_l, grads_weight_perturb_l)]
            grads_u = [a + b.sum(dim=0) for a, b in zip(grads_u, grads_weight_perturb_u)]

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
            grads_weight_perturb_l, grads_weight_perturb_u = ct_utils.grads_helper(
                batch_frag_n, batch_frag_n, label_frag, param_l, param_u, config, False
            )
            grads_n = [a + b.sum(dim=0) for a, b in zip(grads_n, frag_grads_n)]
            grads_l = [a + b.sum(dim=0) for a, b in zip(grads_l, grads_weight_perturb_l)]
            grads_u = [a + b.sum(dim=0) for a, b in zip(grads_u, grads_weight_perturb_u)]
            # apply input transformation
            if transform:
                batch_frag_l, batch_frag_u = transform(batch_frag, model, config.epsilon)
            else:
                batch_frag_l, batch_frag_u = batch_frag - config.epsilon, batch_frag + config.epsilon
            # input + weight perturbed bounds
            grads_input_weight_perturb_l, grads_input_weight_perturb_u = ct_utils.grads_helper(
                batch_frag_l, batch_frag_u, label_frag, param_l, param_u, config, True
            )
            # calculate differences between the input+weight perturbed and weight perturbed bounds
            diffs_l = [a - b for a, b in zip(grads_input_weight_perturb_l, grads_weight_perturb_l)]  # -ve
            diffs_u = [a - b for a, b in zip(grads_input_weight_perturb_u, grads_weight_perturb_u)]  # +ve

            # accumulate and store the the top-k diffs from each fragment
            for i in range(len(grads_n)):
                grads_diffs_l[i].append(torch.topk(diffs_l[i], k_poison, dim=0, largest=False)[0])
                grads_diffs_u[i].append(torch.topk(diffs_u[i], k_poison, dim=0)[0])

        # accumulate the top-k diffs from each fragment then add the overall top-k diffs to the gradient bounds
        grads_diffs_l = [torch.cat(g, dim=0) for g in grads_diffs_l]
        grads_diffs_u = [torch.cat(g, dim=0) for g in grads_diffs_u]
        for i in range(len(grads_n)):
            grads_l[i] += torch.topk(grads_diffs_l[i], k_poison, dim=0, largest=False)[0].sum(dim=0)
            grads_u[i] += torch.topk(grads_diffs_u[i], k_poison, dim=0)[0].sum(dim=0)
            interval_arithmetic.validate_interval(grads_l[i], grads_u[i], grads_n[i])

        # normalise each by the batchsize
        grads_l = [g / batchsize for g in grads_l]
        grads_n = [g / batchsize for g in grads_n]
        grads_u = [g / batchsize for g in grads_u]

        param_n, param_l, param_u = optimizer.step(param_n, param_l, param_u, grads_n, grads_l, grads_u)

    network_eval = config.test_loss_fn(param_n, param_l, param_u, dl_test, model, transform)
    LOGGER.info("Final network eval: %s", ct_utils.get_progress_message(network_eval, param_l, param_u))

    LOGGER.info("=================== Finished Poison Certified Training ===================")

    return param_l, param_n, param_u
