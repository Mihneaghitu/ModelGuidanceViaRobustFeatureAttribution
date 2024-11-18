import os
import sys
sys.path.append("../")

import torch
from models.pipeline import (train_model_with_certified_input_grad, test_model_accuracy, test_delta_input_robustness,
                             load_params_or_results_from_file, write_results_to_file, uniformize_magnitudes_schedule,
                             train_model_with_pgd_robust_input_grad, train_model_with_smoothed_input_grad)
from datasets import derma_mnist, plant, decoy_mnist
from metrics import worst_label_acc
from models.R4_models import DermaNet, PlantNet
from models.fully_connected import FCNAugmented

def ablate(dset_name: str, seed: int, has_conv: bool, criterion: torch.nn.Module, device: torch.device, model_archs: any, size_names: list[str],
    methods: list[str] = ["r4", "ibp_ex", "ibp_ex+r3", "r3"], write_to_file: bool = False, load: bool = False) -> None:
    root_dir = f"saved_experiment_models/ablations/size/{dset_name}/"
    os.makedirs(root_dir, exist_ok=True)
    for method in methods:
        root_dir_method = root_dir + method.removesuffix("_pmo") + "/"
        os.makedirs(root_dir_method, exist_ok=True)
        # Load the params
        params_dict = load_params_or_results_from_file(f"experiment_results/{dset_name}_params.yaml", method)
        delta_threshold = params_dict["delta_threshold"]
        epsilon = params_dict["epsilon"]
        k = params_dict["k"]
        weight_coeff = params_dict["weight_coeff"]
        num_epochs = params_dict["num_epochs"]
        lr = params_dict["lr"]
        restarts = params_dict["restarts"]
        # Manipulate masks based on the dataset
        new_dl_train = dl_train
        if dset_name == "plant":
            if method == "r4":
                new_dl_train = plant.make_soft_masks(new_dl_train, params_dict["alpha_soft"])
        for size_name, arch in zip(size_names, model_archs):
            size_name_dir = f"{root_dir_method}{size_name}"
            os.makedirs(size_name_dir, exist_ok=True)
            train_acc, test_acc, num_robust, min_robust_delta, min_lower_bound, max_upper_bound = 0, 0, 0, 1e+8, 0, 0
            for i in range(restarts):
                torch.manual_seed(i + seed)
                curr_model, loss_fn, multi_class, class_weights = None, "binary_cross_entropy", False, None
                if dset_name == "derma_mnist":
                    curr_model = DermaNet(*arch).to(device)
                    class_weights = [0.4, 5.068] if method not in ["r3"] else None
                else:
                    curr_model = FCNAugmented(*arch).to(device)
                    multi_class = True
                    loss_fn = "cross_entropy"
                print(f"========== Training {dset_name} model with method {method} restart {i} and arch {arch} ==========")
                k_schedule = uniformize_magnitudes_schedule if method == "r3" else None
                if not load:
                    if method in ["pgd_r4", "pgd_r4_pmo"]:
                        train_model_with_pgd_robust_input_grad(new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method, k, device,
                            weight_reg_coeff=weight_coeff, suppress_tqdm=True, class_weights=class_weights)
                    elif method in ["rand_r4", "rand_r4_pmo", "smooth_r3", "smooth_r3_pmo"]:
                        train_model_with_smoothed_input_grad(new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method, k, device,
                            weight_reg_coeff=weight_coeff, suppress_tqdm=True, class_weights=class_weights)
                    else:
                        train_model_with_certified_input_grad(new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method,
                            k, device, has_conv, weight_reg_coeff=weight_coeff, k_schedule=k_schedule, suppress_tqdm=True, class_weights=class_weights)
                else:
                    curr_model.load_state_dict(torch.load(f"{size_name_dir}/run_{i}.pt"))
                    curr_model = curr_model.to(device)
                train_acc += test_model_accuracy(curr_model, new_dl_train, device, multi_class=multi_class, suppress_log=True)
                test_acc += test_model_accuracy(curr_model, dl_test, device, multi_class=multi_class, suppress_log=False)
                n_r, min_delta, m_l, m_u = test_delta_input_robustness(dl_test, curr_model, epsilon, delta_threshold,
                    loss_fn, device, has_conv=has_conv, suppress_log=True)
                num_robust += n_r
                min_robust_delta = min(min_robust_delta, min_delta)
                min_lower_bound += m_l
                max_upper_bound += m_u
                # Save the model
                if not load and write_to_file:
                    torch.save(curr_model.state_dict(), f"{size_name_dir}/run_{i}.pt")
            empty_model, num_classes = None, 2
            if dset_name == "derma_mnist":
                empty_model = DermaNet(*arch).to(device)
            else:
                empty_model = FCNAugmented(*arch).to(device)
                num_classes = 10
            wg_acc, wg = worst_label_acc(empty_model, dl_test, device, num_classes, size_name_dir, suppress_log=True)
            if write_to_file:
                write_results_to_file(f"experiment_results/{dset_name}_size.yaml",
                                        {"train_acc": round(train_acc / restarts, 3),
                                         "test_acc": round(test_acc / restarts, 3),
                                         "worst_group_acc": round(wg_acc, 3),
                                         "worst_group": wg,
                                         "num_robust": round(num_robust / restarts, 3),
                                         "min_lower_bound": round(min_lower_bound / restarts, 3),
                                         "max_upper_bound": round(max_upper_bound / restarts, 3),
                                         "min_robust_delta": min_robust_delta}, method + f"_{size_name}")

assert len(sys.argv) == 2
assert sys.argv[1] in ["derma_mnist", "decoy_mnist"]
if sys.argv[1] == "derma_mnist":
    IMG_SIZE = 64
    train_dset = derma_mnist.DecoyDermaMNIST(True, size=IMG_SIZE)
    test_dset = derma_mnist.DecoyDermaMNIST(False, size=IMG_SIZE)
    dl_train, dl_test = derma_mnist.get_dataloader(train_dset, 256), derma_mnist.get_dataloader(test_dset, 256)
    dev = torch.device("cuda:0")
    archs = [(3, IMG_SIZE, 1, "small"), (3, IMG_SIZE, 1, "small_medium"), (3, IMG_SIZE, 1), (3, IMG_SIZE, 1, "large")]
    sz_ns = ["small", "small_medium", "medium_large", "large"]
    ablate(sys.argv[1], 0, True, torch.nn.BCELoss(), dev, archs, sz_ns, write_to_file=True, methods=["ibp_ex+r3", "r3", "r4", "rand_r4_pmo"])
elif sys.argv[1] == "decoy_mnist":
    dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
    dl_train, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
    dev = torch.device("cuda:0")
    archs = [(784, 10, 512, 1), (784, 10, 512, 2), (784, 10, 512, 3), (784, 10, 512, 4)]
    sz_ns = ["1_layer", "2_layer", "3_layer", "4_layer"]
    ablate(sys.argv[1], 0, False, torch.nn.CrossEntropyLoss(), dev, archs, sz_ns, write_to_file=True,
           methods=["r4", "r3", "ibp_ex+r3", "rand_r4"])
else:
    raise ValueError("Only 'derma_mnist' and 'plant' are supported")
