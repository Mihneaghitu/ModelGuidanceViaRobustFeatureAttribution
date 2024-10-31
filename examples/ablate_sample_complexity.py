import os
import sys
sys.path.append("../")

import torch
from models.pipeline import (train_model_with_certified_input_grad, test_model_accuracy,
                             test_delta_input_robustness, load_params_or_results_from_file,
                             write_results_to_file, uniformize_magnitudes_schedule)
from datasets import derma_mnist, plant, decoy_mnist
from models.R4_models import DermaNet, PlantNet
from models.fully_connected import FCNAugmented

def ablate(dset_name: str, seed: int, has_conv: bool, criterion: torch.nn.Module, device: torch.device,
    methods: list[str] = ["r4", "ibp_ex", "ibp_ex+r3", "r3"], img_size: int = None, with_data_removal: bool = False) -> None:
    suffix = "" if  not with_data_removal else "_data_removal"
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
            r4_soft = (method == "r4" and (not with_data_removal))
            if dset_name == "derma_mnist":
                new_dl_train = derma_mnist.remove_masks(mask_ratio, dl_train, with_data_removal, r4_soft)
            elif dset_name == "plant":
                new_dl_train = plant.remove_masks(mask_ratio, dl_train, with_data_removal, r4_soft)
                if method == "r4":
                    new_dl_train = plant.make_soft_masks(new_dl_train, params_dict["alpha_soft"])
            else:
                new_dl_train = decoy_mnist.remove_masks(mask_ratio, dl_train, with_data_removal, r4_soft)
            train_acc, test_acc, num_robust, min_robust_delta, min_lower_bound, max_upper_bound = 0, 0, 0, 1e+8, 0, 0
            for i in range(restarts):
                torch.manual_seed(i + seed)
                curr_model, loss_fn, multi_class = "binary_cross_entropy", None, False
                if dset_name == "derma_mnist":
                    curr_model = DermaNet(3, img_size, 1)
                    curr_model = torch.nn.DataParallel(curr_model, device_ids=[1, 0])
                elif dset_name == "plant":
                    curr_model = PlantNet(3, 1)
                    curr_model = torch.nn.DataParallel(curr_model, device_ids=[1, 0])
                else:
                    curr_model = FCNAugmented(784, 10, 512, 1)
                    curr_model = curr_model.to(device)
                    multi_class = True
                    loss_fn = "cross_entropy"
                print(f"========== Training model with method {method} restart {i} and mask ratio {mask_ratio} ==========")
                k_schedule = uniformize_magnitudes_schedule if method == "r3" else None
                train_model_with_certified_input_grad(new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method,
                    k, device, has_conv, weight_reg_coeff=weight_coeff, k_schedule=k_schedule, suppress_tqdm=True)
                train_acc += test_model_accuracy(curr_model, new_dl_train, device, multi_class=multi_class, suppress_log=True)
                test_acc += test_model_accuracy(curr_model, dl_test, device, multi_class=multi_class, suppress_log=True)
                n_r, min_delta, m_l, m_u = test_delta_input_robustness(dl_test, curr_model, epsilon, delta_threshold,
                    loss_fn, device, has_conv=has_conv, suppress_log=True)
                num_robust += n_r
                min_robust_delta = min(min_robust_delta, min_delta)
                min_lower_bound += m_l
                max_upper_bound += m_u
            write_results_to_file(f"experiment_results/{dset_name}_sample_complexity" + suffix + ".yaml",
                                {"train_acc": round(train_acc / restarts, 3),
                                 "test_acc": round(test_acc / restarts, 3),
                                 "num_robust": round(num_robust / restarts, 3),
                                 "min_lower_bound": round(min_lower_bound / restarts, 3),
                                 "max_upper_bound": round(max_upper_bound / restarts, 3),
                                 "min_robust_delta": min_robust_delta}, method + f"_{int(mask_ratio * 100)}")

assert len(sys.argv) == 3
assert sys.argv[1] in ["derma_mnist", "plant", "decoy_mnist"]
assert sys.argv[2] in ["0", "1"] # 0 no data removal, 1 with data removal
if sys.argv[1] == "derma_mnist":
    IMG_SIZE = 64
    train_dset = derma_mnist.DecoyDermaMNIST(True, size=IMG_SIZE)
    test_dset = derma_mnist.DecoyDermaMNIST(False, size=IMG_SIZE)
    dl_train, dl_test = derma_mnist.get_dataloader(train_dset, 256), derma_mnist.get_dataloader(test_dset, 256)
    dev = torch.device("cuda:0")
    ablate("derma_mnist", 0, True, torch.nn.BCELoss(), dev, methods=["r3"], img_size=IMG_SIZE, with_data_removal=bool(int(sys.argv[2])))
elif sys.argv[1] == "plant":
    SPLIT_ROOT = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
    DATA_ROOT = "/vol/bitbucket/mg2720/plant/rgb_data"
    MASKS_FILE = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
    plant_train_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, True)
    plant_test_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, False)
    dl_train, dl_test = plant.get_dataloader(plant_train_2, 50), plant.get_dataloader(plant_test_2, 25)
    dev = torch.device("cuda:1")
    ablate("plant", 0, True,  torch.nn.BCELoss(), dev, methods=["r4"], with_data_removal=bool(int(sys.argv[2])))
elif sys.argv[1] == "decoy_mnist":
    SEED = 0
    dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
    dl_train, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
    dev = torch.device("cuda:0")
    ablate("decoy_mnist", 0, False, torch.nn.CrossEntropyLoss(), dev, methods=["r4"], with_data_removal=bool(int(sys.argv[2])))
else:
    raise ValueError("Only 'derma_mnist' and 'plant' are supported")
