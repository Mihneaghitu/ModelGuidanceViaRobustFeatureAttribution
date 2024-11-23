import copy
import os
import sys

sys.path.append("../")

import torch
import wandb
import random
import numpy as np

from models.R4_models import PlantNet
from models.pipeline import (train_model_with_certified_input_grad, train_model_with_pgd_robust_input_grad,
                             train_model_with_smoothed_input_grad, test_model_accuracy, test_delta_input_robustness,
                             test_macro_avg_label_accuracy, test_model_avg_and_wg_accuracy)
from metrics import get_restart_avg_and_worst_group_accuracy_with_stddev
from datasets import plant


SPLIT_ROOT = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
DATA_ROOT = "/vol/bitbucket/mg2720/plant/rgb_data"
MASKS_FILE = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
MODEL_ROOT_SAVE_DIR = "saved_experiment_models/performance/plant"
SEED = 0

plant_train_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, True)
plant_test_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, False)
dl_train = plant.get_dataloader(plant_train_2, 50)
dl_test = plant.get_dataloader(plant_test_2, 10)
device = torch.device("cuda:0")
criterion = torch.nn.BCELoss()
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

os.makedirs(MODEL_ROOT_SAVE_DIR, exist_ok=True)
methods = ["std", "r3", "r4", "ibp_ex", "ibp_ex+r3", "smooth_r3", "rand_r4", "pgd_r4"]
save_dir_for_method = {method: os.path.join(MODEL_ROOT_SAVE_DIR, method) for method in methods}

def default_config(method: str) -> dict:
    sl_grid_config = {
        'method': 'bayes',
        'metric': {
            'name': 'worst_group_accuracy',
            'goal': 'maximize'
        }
    }
    params_dict = {
        "num_epochs": {
            'min': 10,
            'max': 20
        },
        "lr": {
            'min': 0.0001,
            'max': 0.0005
        },
        "class_weight_0": {
            'min': 1.85,
            'max': 4.0
        },
        "restarts": {
            'values': [3]
        },
        "epsilon": {
            'min': 0.01,
            'max': 0.03
        },
        "alpha": {
            'min': 0.675,
            'max': 0.8
        },
        "k": {
            'values': [0.01]
        },
        "method": {
            'values': [method]
        }
    }
    early_stopping = {
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27
    }
    sl_grid_config['parameters'] = params_dict
    sl_grid_config['early_terminate'] = early_stopping

    return sl_grid_config

def bayes_search_sl():
    with wandb.init():
        grid_config = wandb.config
        lr = grid_config.lr
        restarts = grid_config.restarts
        epsilon = grid_config.epsilon
        num_epochs = grid_config.num_epochs
        class_weight_0 = grid_config.class_weight_0
        alpha = grid_config.alpha
        k = grid_config.k
        method = grid_config.method
        class_weights = [class_weight_0, 1]

        # Train standard 3 times and test accuracy and delta input robustness for the masked region
        new_dl_train = plant.make_soft_masks(dl_train, alpha)
        avg_min_delta, avg_lb, avg_ub = 0, 0, 0
        for i in range(restarts):
            # Reinitialize the model
            # We could try to just reinitialize the weights, but we can throw away the previous model for now as we do not need it
            torch.manual_seed(i)
            curr_model = PlantNet(3, 1).to(device)

            print(f"========== Training model with method {method} restart {i} ==========")
            train_model_with_certified_input_grad(
                new_dl_train, num_epochs, curr_model, lr, criterion, epsilon, method, k, device, True, class_weights = class_weights
            )
            _, min_delta, lb, ub = test_delta_input_robustness(
                dl_test, curr_model, grid_config.epsilon, 1, "binary_cross_entropy", device, has_conv=True
            )
            avg_min_delta += min_delta / restarts
            avg_lb += lb / restarts
            avg_ub += ub / restarts
            torch.save(curr_model.state_dict(), os.path.join(save_dir_for_method[method], f"run_{i}.pt"))
        empty_model = PlantNet(3, 1).to(device)
        macro_avg_acc, wg_acc, wg, *_ = get_restart_avg_and_worst_group_accuracy_with_stddev(
            dl_test, save_dir_for_method[method], empty_model, device, num_groups=2
        )
        wandb.log({
            "min_delta": round(avg_min_delta, 5),
            "worst_group_accuracy": round(wg_acc, 5),
            "worst_group": wg,
            "lower_bound": round(lb, 5),
            "upper_bound": round(ub, 5),
            "macro_avg_group_accuracy": round(macro_avg_acc, 5),
        })

def setup():
    wandb.login(key="6af656612e6115c4b189c6074dadbfc436f21439")

def run_r4_sweep():
    setup()
    r4_sweep_config = default_config("r4")

    sweep_id_1 = wandb.sweep(r4_sweep_config, project="r4_plant")
    wandb.agent(sweep_id=sweep_id_1, function=bayes_search_sl, count=50)

def run_ibp_ex_sweep():
    setup()
    ibp_ex_sweep_config = default_config("ibp_ex")
    ibp_ex_sweep_config["parameters"]["weight_reg_coeff"] = {'values': [0.0001]}

    sweep_id_1 = wandb.sweep(ibp_ex_sweep_config, project="ibp_ex_plant")
    wandb.agent(sweep_id=sweep_id_1, function=bayes_search_sl, count=50)

run_r4_sweep()
run_ibp_ex_sweep()
