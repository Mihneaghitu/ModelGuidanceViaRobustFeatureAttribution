import os
import sys
import inspect
sys.path.append("../")

import torch
from models.pipeline import (train_model_with_certified_input_grad, load_params_or_results_from_file, write_results_to_file,
                             train_model_with_pgd_robust_input_grad, train_model_with_smoothed_input_grad)
from models.R4_models import DermaNet
from models.fully_connected import FCNAugmented
from datasets import derma_mnist, decoy_mnist
from metrics import get_restart_avg_and_worst_group_accuracy_with_stddev, get_avg_rob_metrics

def ablate(dset_name: str,
           train_dloader: torch.utils.data.DataLoader,
           test_dloader: torch.utils.data.DataLoader,
           train_func: dict[str, callable],
           device: torch.device,
           model_archs: list[float],
           size_names: list[str],
           methods: list[str],
           write_to_file: bool = False) -> None:
    root_dir = f"saved_experiment_models/ablations/size/{dset_name}/"
    os.makedirs(root_dir, exist_ok=True)
    for method in methods:
        os.makedirs(root_dir + method, exist_ok=True)
        # Load the params
        params_dict = load_params_or_results_from_file(f"experiment_results/{dset_name}_params.yaml", method)
        class_weights = params_dict["class_weights"]
        if not isinstance(class_weights, list):
            class_weights = None
        test_epsilon = params_dict["test_epsilon"]
        multi_class = params_dict["multi_class"]
        num_epochs = params_dict["num_epochs"]
        has_conv = params_dict["has_conv"]
        restarts = params_dict["restarts"]
        epsilon = params_dict["epsilon"]
        lr = params_dict["lr"]
        k = params_dict["k"]
        # optional ones
        weight_decay = 0
        weight_coeff = 0
        if "weight_decay" in params_dict and params_dict["weight_decay"] > 0:
            weight_decay = params_dict["weight_decay"]
        if "weight_coeff" in params_dict and params_dict["weight_coeff"] > 0:
            weight_coeff = params_dict["weight_coeff"]
        for size_name, arch in zip(size_names, model_archs):
            size_name_dir = root_dir + method + f"/{size_name}"
            os.makedirs(size_name_dir, exist_ok=True)
            # Manipulate masks based on the dataset
            for i in range(restarts):
                # Seed 0
                torch.manual_seed(i)
                curr_model, loss_fn, criterion = None, "binary_cross_entropy", torch.nn.BCELoss()
                if dset_name == "decoy_mnist":
                    curr_model = FCNAugmented(*arch).to(device)
                    loss_fn, criterion = "cross_entropy", torch.nn.CrossEntropyLoss()
                else:
                    curr_model = DermaNet(*arch).to(device)
                print(f"========== Training model with method {method} restart {i} and arch {size_name} ==========")
                #! The certified training function has an additional argument "has_conv" which is not present in the smoothed or pgd version
                #! I want to do this general, though, so the solution is to inspect the function signature, see if it has the argument,
                #! and pass it to the required arguments tuple if that is the case (thus, the arguments are built dynamically)
                #! I think this handles it gracefully.
                sig = inspect.signature(train_func[method])
                required_args = train_dloader, num_epochs, curr_model, lr, criterion, epsilon, method, k, device
                if sig.parameters.get("has_conv"):
                    required_args = (*required_args, has_conv)
                train_func[method](*required_args, weight_reg_coeff=weight_coeff, weight_decay=weight_decay, suppress_tqdm=True, class_weights=class_weights)
                # Save the model
                torch.save(curr_model.state_dict(), f"{size_name_dir}/run_{i}.pt")
            empty_model, num_groups = None, None
            if dset_name == "decoy_mnist":
                empty_model = FCNAugmented(*arch).to(device)
                num_groups = 10
            else:
                empty_model = DermaNet(*arch).to(device)
                num_groups = 2
            #* Measure (core and spurious) accuracy metrics
            macro_avg_acc, wg_acc, wg, stddev_group_acc, stddev_wg_acc, acc_per_group, stddev_per_group = get_restart_avg_and_worst_group_accuracy_with_stddev(
                test_dloader, size_name_dir, empty_model, device, num_groups, multi_class=multi_class, suppress_log=True, return_stddev_per_group=True
            )
            #* Measure robustness metrics
            delta_mean, ls_mean, us_mean, delta_std, *_ = get_avg_rob_metrics(
                empty_model, test_dloader, device, size_name_dir, test_epsilon, loss_fn, has_conv=has_conv
            )
            delta_mean, ls_mean, us_mean, delta_std = round(float(delta_mean.item()), 5), round(float(ls_mean.item()), 5), round(float(us_mean.item()), 5), round(float(delta_std.item()), 5)
            if write_to_file:
                write_results_to_file(f"experiment_results/{dset_name}_size.yaml",
                      {"avg_group_acc": round(macro_avg_acc, 5),
                       "worst_group_acc": round(wg_acc, 5),
                       "worst_group": wg,
                       "stddev_acc": stddev_group_acc,
                       "stddev_per_group": stddev_per_group,
                       "acc_per_group": acc_per_group,
                       "stddev_worst_group_acc": stddev_wg_acc,
                       "lb_mean": round(ls_mean, 5),
                       "ub_mean": round(us_mean, 5),
                       "delta_mean": round(delta_mean, 5),
                       "delta_stddev": round(delta_std, 5),
                       }, method + f"_{size_name}")

assert len(sys.argv) == 3, "Usage: python ablate_net_size.py <derma_mnist|decoy_mnist> <device>"
assert sys.argv[1] in ["decoy_mnist", "derma_mnist"], "Only 'decoy_mnist' and 'derma_mnist' are supported"
assert sys.argv[2] in ["d0", "d1"], "Only 'd0' and 'd1' are supported, d0 mean cuda:0, d1 means cuda:1"
funcs = {
    "r4": train_model_with_certified_input_grad,
    "ibp_ex": train_model_with_certified_input_grad,
    "ibp_ex+r3": train_model_with_certified_input_grad,
    "pgd_r4": train_model_with_pgd_robust_input_grad,
    "rand_r4": train_model_with_smoothed_input_grad,
}
dev = torch.device("cuda:" + sys.argv[2][-1])
mlx_methods = ["rand_r4"]# ["r3", "ibp_ex", "r4", "pgd_r4", "rand_r4"]
if sys.argv[1] == "decoy_mnist":
    funcs["r3"] = train_model_with_certified_input_grad
    dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
    dl_train, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
    archs = [(784, 10, 512, 1), (784, 10, 512, 2), (784, 10, 512, 3), (784, 10, 512, 4)]
    sz_ns = ["1_layer", "2_layer", "3_layer", "4_layer"]
    ablate(sys.argv[1], dl_train, dl_test, funcs, dev, archs, sz_ns, mlx_methods, write_to_file=True)
else:
    IMG_SIZE = 64
    funcs["r3"] = train_model_with_pgd_robust_input_grad
    train_dset = derma_mnist.DecoyDermaMNIST(True, size=IMG_SIZE)
    test_dset = derma_mnist.DecoyDermaMNIST(False, size=IMG_SIZE)
    dl_train, dl_test = derma_mnist.get_dataloader(train_dset, 256), derma_mnist.get_dataloader(test_dset, 100)
    archs = [(3, IMG_SIZE, 1, "small"), (3, IMG_SIZE, 1, "small_medium"), (3, IMG_SIZE, 1), (3, IMG_SIZE, 1, "large")]
    sz_ns = ["small", "small_medium", "medium_large", "large"]
    ablate(sys.argv[1], dl_train, dl_test, funcs, dev, archs, sz_ns, mlx_methods, write_to_file=True)
