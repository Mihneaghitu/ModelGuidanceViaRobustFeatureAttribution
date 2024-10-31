import os
import sys
sys.path.append("../")

import torch
from models.pipeline import (train_model_with_certified_input_grad, test_model_accuracy, train_model_with_pgd_robust_input_grad,
                             test_delta_input_robustness, load_params_or_results_from_file, write_results_to_file, uniformize_magnitudes_schedule)
from datasets import derma_mnist, plant
from models.R4_models import DermaNet, PlantNet

def ablate(dset_name: str, methods: list[str] = ["r4", "ibp_ex", "ibp_ex+r3", "r3"]) -> None:
    mask_ratios = [0.8, 0.6, 0.4, 0.2]
    for method in methods:
        # Load the params
        params_dict = load_params_or_results_from_file(f"experiment_results/{dset_name}_params.yaml", method)
        delta_threshold = params_dict["delta_threshold"]
        epsilon = params_dict["epsilon"]
        k = params_dict["k"]
        weight_coeff = params_dict["weight_coeff"]
        num_epochs = params_dict["num_epochs"]
        lr = params_dict["lr"]
        restarts = params_dict["restarts"]
        for mask_ratio in mask_ratios:
            # Manipulate masks based on the dataset
            new_dl_train = None
            if dset_name == "derma_mnist":
                new_dl_train = derma_mnist.remove_masks(mask_ratio, dl_train)
            if dset_name == "plant":
                new_dl_train = plant.remove_masks(mask_ratio, dl_train)
                if method == "r4":
                    new_dl_train = plant.make_soft_masks(new_dl_train, params_dict["alpha_soft"])
            train_acc, test_acc, num_robust, min_robust_delta, min_lower_bound, max_upper_bound = 0, 0, 0, 1e+8, 0, 0
            for i in range(restarts):
                # Reinitialize the model
                # We could try to just reinitialize the weights, but we can throw away the previous model for now as we do not need it
                torch.manual_seed(i + SEED)
                curr_model = DermaNet(3, IMG_SIZE, 1) if dset_name == "derma_mnist" else PlantNet(3, 1)
                curr_model = torch.nn.DataParallel(curr_model, device_ids=[1, 0] if dset_name == "plant" else [0])

                print(f"========== Training model with method {method} restart {i} and mask ratio {mask_ratio} ==========")
                k_schedule = uniformize_magnitudes_schedule if method == "r3" else None
                train_model_with_certified_input_grad(new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method,
                    k, device, True, weight_reg_coeff=weight_coeff, k_schedule=k_schedule, suppress_tqdm=True)
                train_acc += test_model_accuracy(curr_model, new_dl_train, device, suppress_log=True)
                test_acc += test_model_accuracy(curr_model, dl_test, device, suppress_log=True)
                n_r, min_delta, m_l, m_u = test_delta_input_robustness(dl_test, curr_model, epsilon, delta_threshold,
                                                             "binary_cross_entropy", device, has_conv=True, suppress_log=True)
                num_robust += n_r
                min_robust_delta = min(min_robust_delta, min_delta)
                min_lower_bound += m_l
                max_upper_bound += m_u
            write_results_to_file(f"experiment_results/{dset_name}_sample_complexity.yaml",
                                {"train_acc": round(train_acc / restarts, 3),
                                 "test_acc": round(test_acc / restarts, 3),
                                 "num_robust": round(num_robust / restarts, 3),
                                 "min_lower_bound_avg": round(min_lower_bound / restarts, 3),
                                 "max_upper_bound_avg": round(max_upper_bound / restarts, 3),
                                 "min_robust_delta": min_robust_delta}, method + f"_{int(mask_ratio * 100)}")

if sys.argv[1] == "derma_mnist":
    SEED = 0
    IMG_SIZE = 64
    train_dset = derma_mnist.DecoyDermaMNIST(True, size=IMG_SIZE)
    test_dset = derma_mnist.DecoyDermaMNIST(False, size=IMG_SIZE)
    dl_train, dl_test = derma_mnist.get_dataloader(train_dset, 256), derma_mnist.get_dataloader(test_dset, 256)
    criterion = torch.nn.BCELoss()
    device = torch.device("cuda:0")
    ablate("derma_mnist", ["r3"])
elif sys.argv[1] == "plant":
    SPLIT_ROOT = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
    DATA_ROOT = "/vol/bitbucket/mg2720/plant/rgb_data"
    MASKS_FILE = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
    plant_train_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, True)
    plant_test_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, False)
    SEED = 0
    dl_train, dl_test = plant.get_dataloader(plant_train_2, 50), plant.get_dataloader(plant_test_2, 25)
    criterion = torch.nn.BCELoss()
    device = torch.device("cuda:1")
    ablate("plant", ["ibp_ex", "ibp_ex+r3", "r3"])
else:
    raise ValueError("Only 'derma_mnist' and 'plant' are supported")
