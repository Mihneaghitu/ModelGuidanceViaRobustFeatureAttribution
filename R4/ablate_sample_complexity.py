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

def make_hmap(dset_name: str,
           train_dloader: torch.utils.data.DataLoader,
           test_dloader: torch.utils.data.DataLoader,
           train_func: dict[str, callable],
           device: torch.device,
           mask_ratios: list[float],
           methods: list[str],
           with_data_removal: bool = False,
           write_to_file: bool = False) -> None:
    root_dir = f"saved_experiment_models/ablations/hmap/{dset_name}/"
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
        wd_init = weight_coeff if weight_coeff > 0 else weight_decay
        wd_ratios = [wd_init / 1000, wd_init / 100, wd_init / 10, wd_init, wd_init * 10, wd_init * 100]
        #* Hardcode this special case because R4 DecoyMNIST best model has 0 weight regularization
        if method == "r3" and dset_name == "decoy_mnist":
            wd_ratios = [0.001, 0.01, 0.1, 0, 1, 10]
        for idx_wd, wd in enumerate(wd_ratios):
            if "weight_coeff" in params_dict and params_dict["weight_coeff"] > 0:
                weight_coeff = wd
            else:
                weight_decay = wd
            for mask_ratio in mask_ratios:
                mask_and_wd_ratio_dir = root_dir + method + f"/wd_{idx_wd}" + f"/ratio_{int(mask_ratio * 100)}"
                os.makedirs(mask_and_wd_ratio_dir, exist_ok=True)
                # Manipulate masks based on the dataset
                new_dl_train = None
                if dset_name == "derma_mnist":
                    new_dl_train = derma_mnist.remove_masks(mask_ratio, train_dloader, with_data_removal)
                else:
                    new_dl_train = decoy_mnist.remove_masks(mask_ratio, train_dloader, with_data_removal)
                for i in range(restarts):
                    # Seed 0
                    torch.manual_seed(i)
                    curr_model, loss_fn, criterion = None, "binary_cross_entropy", torch.nn.BCELoss()
                    if dset_name == "derma_mnist":
                        curr_model = DermaNet(3, 64, 1).to(device)
                    else:
                        curr_model = FCNAugmented(784, 10, 512, 1).to(device)
                        loss_fn, criterion = "cross_entropy", torch.nn.CrossEntropyLoss()
                    print(f"========== Training model with method {method} restart {i} and mask ratio {mask_ratio} ==========")
                    #! The certified training function has an additional argument "has_conv" which is not present in the smoothed or pgd version
                    #! I want to do this general, though, so the solution is to inspect the function signature, see if it has the argument,
                    #! and pass it to the required arguments tuple if that is the case (thus, the arguments are built dynamically)
                    #! I think this handles it gracefully.
                    sig = inspect.signature(train_func[method])
                    required_args = new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method, k, device
                    if sig.parameters.get("has_conv"):
                        required_args = (*required_args, has_conv)
                    train_func[method](*required_args, weight_reg_coeff=weight_coeff, weight_decay=weight_decay, suppress_tqdm=True, class_weights=class_weights)
                    # Save the model
                    torch.save(curr_model.state_dict(), f"{mask_and_wd_ratio_dir}/run_{i}.pt")
                empty_model, num_groups = None, None
                if dset_name == "derma_mnist":
                    empty_model = DermaNet(3, 64, 1).to(device)
                    num_groups = 2
                else:
                    empty_model = FCNAugmented(784, 10, 512, 1).to(device)
                    num_groups = 10
                #* Measure (core and spurious) accuracy metrics
                macro_avg_acc, wg_acc, wg, stddev_group_acc, stddev_wg_acc, acc_per_group, stddev_per_group = get_restart_avg_and_worst_group_accuracy_with_stddev(
                    test_dloader, mask_and_wd_ratio_dir, empty_model, device, num_groups, multi_class=multi_class, suppress_log=True, return_stddev_per_group=True
                )
                #* Measure robustness metrics
                delta_mean, ls_mean, us_mean, delta_std, *_ = get_avg_rob_metrics(
                    empty_model, test_dloader, device, mask_and_wd_ratio_dir, test_epsilon, loss_fn, has_conv=has_conv
                )
                delta_mean, ls_mean, us_mean, delta_std = round(float(delta_mean.item()), 5), round(float(ls_mean.item()), 5), round(float(us_mean.item()), 5), round(float(delta_std.item()), 5)
                if write_to_file:
                    write_results_to_file(f"experiment_results/{dset_name}_hmap.yaml",
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
                           }, method + f"wd_{idx_wd}_{int(mask_ratio * 100)}")

def ablate(dset_name: str,
           train_dloader: torch.utils.data.DataLoader,
           test_dloader: torch.utils.data.DataLoader,
           train_func: dict[str, callable],
           device: torch.device,
           mask_ratios: list[float],
           methods: list[str],
           with_data_removal: bool = False,
           decrease_l2_strength: bool = False,
           write_to_file: bool = False) -> None:
    suffix = "" if not with_data_removal else "_data_removal"
    suffix = suffix + "_propl2" if decrease_l2_strength else suffix
    ablation_type = "data_and_mask" if with_data_removal else "mask"
    ablation_type = ablation_type + "_propl2" if decrease_l2_strength else ablation_type
    root_dir = f"saved_experiment_models/ablations/{ablation_type}/{dset_name}/"
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
        num_samples = None
        if "weight_decay" in params_dict and params_dict["weight_decay"] > 0:
            weight_decay = params_dict["weight_decay"]
        if "weight_coeff" in params_dict and params_dict["weight_coeff"] > 0:
            weight_coeff = params_dict["weight_coeff"]
        if "num_samples" in params_dict:
            num_samples = params_dict["num_samples"]
        for mask_ratio in mask_ratios:
            if decrease_l2_strength:
                weight_coeff = weight_coeff * mask_ratio
                weight_decay = weight_decay * mask_ratio
            mask_ratio_dir = root_dir + method + f"/ratio_{int(mask_ratio * 100)}"
            os.makedirs(mask_ratio_dir, exist_ok=True)
            # Manipulate masks based on the dataset
            new_dl_train = None
            if dset_name == "derma_mnist":
                new_dl_train = derma_mnist.remove_masks(mask_ratio, train_dloader, with_data_removal)
            else:
                new_dl_train = decoy_mnist.remove_masks(mask_ratio, train_dloader, with_data_removal)
            for i in range(restarts):
                # Seed 0
                torch.manual_seed(i)
                curr_model, loss_fn, criterion = None, "binary_cross_entropy", torch.nn.BCELoss()
                if dset_name == "derma_mnist":
                    curr_model = DermaNet(3, 64, 1).to(device)
                else:
                    curr_model = FCNAugmented(784, 10, 512, 1).to(device)
                    loss_fn, criterion = "cross_entropy", torch.nn.CrossEntropyLoss()
                print(f"========== Training model with method {method} restart {i} and mask ratio {mask_ratio} ==========")
                #! The certified training function has an additional argument "has_conv" which is not present in the smoothed or pgd version
                #! I want to do this general, though, so the solution is to inspect the function signature, see if it has the argument,
                #! and pass it to the required arguments tuple if that is the case (thus, the arguments are built dynamically)
                #! I think this handles it gracefully.
                sig = inspect.signature(train_func[method])
                required_args = new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method, k, device
                if sig.parameters.get("has_conv"):
                    required_args = (*required_args, has_conv)
                if num_samples is not None:
                    train_func[method](*required_args, num_samples=num_samples, weight_reg_coeff=weight_coeff, weight_decay=weight_decay, suppress_tqdm=True, class_weights=class_weights)
                else:
                    train_func[method](*required_args, weight_reg_coeff=weight_coeff, weight_decay=weight_decay, suppress_tqdm=True, class_weights=class_weights)
                # Save the model
                torch.save(curr_model.state_dict(), f"{mask_ratio_dir}/run_{i}.pt")
            empty_model, num_groups = None, None
            if dset_name == "derma_mnist":
                if decrease_l2_strength:
                    weight_coeff = weight_coeff * 0.1 # i.e. the proportionality constant is mask_ratio * 0.1
                    weight_decay = weight_decay * 0.1 # i.e. the proportionality constant is mask_ratio * 0.1
                empty_model = DermaNet(3, 64, 1).to(device)
                num_groups = 2
            else:
                empty_model = FCNAugmented(784, 10, 512, 1).to(device)
                num_groups = 10
            #* Measure (core and spurious) accuracy metrics
            macro_avg_acc, wg_acc, wg, stddev_group_acc, stddev_wg_acc, acc_per_group, stddev_per_group = get_restart_avg_and_worst_group_accuracy_with_stddev(
                test_dloader, mask_ratio_dir, empty_model, device, num_groups, multi_class=multi_class, suppress_log=True, return_stddev_per_group=True
            )
            #* Measure robustness metrics
            delta_mean, ls_mean, us_mean, delta_std, *_ = get_avg_rob_metrics(
                empty_model, test_dloader, device, mask_ratio_dir, test_epsilon, loss_fn, has_conv=has_conv
            )
            delta_mean, ls_mean, us_mean, delta_std = round(float(delta_mean.item()), 5), round(float(ls_mean.item()), 5), round(float(us_mean.item()), 5), round(float(delta_std.item()), 5)
            if write_to_file:
                write_results_to_file(f"experiment_results/{dset_name}_sample_complexity" + suffix + ".yaml",
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
                       }, method + f"_{int(mask_ratio * 100)}")

# def test():
#     sys.argv = ["", "decoy_mnist", "0", "d0"]
assert len(sys.argv) == 6, "Usage: python ablate_sample_complexity.py <dataset> <remove_data> <device> <decrease_l2_strength> <heatmap>"
assert sys.argv[1] in ["derma_mnist", "decoy_mnist"], "Only 'derma_mnist' and 'decoy_mnist' are supported"
assert sys.argv[2] in ["0", "1"], "Only 0 and 1 are supported, 0 means only masks are removed, 1 means both masks and data are removed"
assert sys.argv[3] in ["d0", "d1"], "Only d0 and d1 are supported, d0 means cuda:0, d1 means cuda:1"
assert sys.argv[4] in ["dl2", "ndl2"], "Only dl2 and ndl2 are supported, dl2 means proportional decrease in l2 strength, ndl2 means no decrease"
assert sys.argv[5] in ["hm", "nhm"], "Only hm and nhm are supported, hm means heatmaps are generated, nhm means no heatmaps are generated"
#* General setup
funcs = {
    "r4": train_model_with_certified_input_grad,
    "ibp_ex": train_model_with_certified_input_grad,
    "ibp_ex+r3": train_model_with_certified_input_grad,
    "pgd_r4": train_model_with_pgd_robust_input_grad,
    "rand_r4": train_model_with_smoothed_input_grad,
}
dev = torch.device("cuda:" + sys.argv[3][-1])
# mask ratios
mrs = [0.8, 0.6, 0.4, 0.2]
mlx_methods = ["r3", "ibp_ex", "r4", "pgd_r4", "rand_r4"]
if not bool(int(sys.argv[2])): # only masks are removed
    mrs.append(0)
remove_data = bool(int(sys.argv[2]))
dl2 = sys.argv[4] == "dl2"
hm = sys.argv[5] == "hm"
if hm:
    mrs.insert(0, 1)
    mlx_methods.append("ibp_ex+r3")
#* Specific dataset setup
match sys.argv[1]:
    case "decoy_mnist":
        funcs["r3"] = train_model_with_certified_input_grad
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        dl_train, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        if hm:
            make_hmap("decoy_mnist", dl_train, dl_test, funcs, dev, mrs, mlx_methods, write_to_file=True)
        else:
            ablate("decoy_mnist", dl_train, dl_test, funcs, dev, mrs, mlx_methods,
                   write_to_file=True, with_data_removal=remove_data, decrease_l2_strength=dl2)
    case "derma_mnist":
        funcs["r3"] = train_model_with_pgd_robust_input_grad
        train_dset = derma_mnist.DecoyDermaMNIST(True, size=64)
        test_dset = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_train, dl_test = derma_mnist.get_dataloader(train_dset, 256), derma_mnist.get_dataloader(test_dset, 100)
        if hm:
            make_hmap("derma_mnist", dl_train, dl_test, funcs, dev, mrs, mlx_methods, write_to_file=True)
        else:
            ablate("derma_mnist", dl_train, dl_test, funcs, dev, mrs, mlx_methods,
                   write_to_file=True, with_data_removal=remove_data, decrease_l2_strength=dl2)
    case _:
        raise ValueError("Only 'decoy_mnist' and 'derma_mnist' are supported")
