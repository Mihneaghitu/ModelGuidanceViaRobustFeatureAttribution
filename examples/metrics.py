import os
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from datasets import derma_mnist, isic, plant, decoy_mnist, salient_imagenet
from models.R4_models import DermaNet, PlantNet, SalientImageNet, LesionNet
from models.fully_connected import FCNAugmented
from models.pipeline import test_delta_input_robustness, test_model_accuracy
import numpy as np
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from math import fabs


def get_restart_avg_and_worst_group_accuracy_with_stddev(
    dl_test_grouped: torch.utils.data.DataLoader, model_run_dir: str, model: torch.nn.Sequential, device: str, num_groups: int,
    multi_class: bool = False, suppress_log: bool = False) -> tuple[float, float, int, float, float]:
    restarts = len(os.listdir(model_run_dir))
    acc_per_group, num_elems_for_group = np.zeros((restarts, num_groups)), np.zeros((restarts, num_groups))
    for run in range(restarts):
        run_file = os.path.join(model_run_dir, f"run_{run}.pt")
        model.load_state_dict(torch.load(run_file))
        model = model.to(device)
        model.eval()
        for data, ground_truth_labels, _, groups in dl_test_grouped:
            data, ground_truth_labels, groups = data.to(device), ground_truth_labels.to(device), groups.to(device)
            output = model(data).squeeze()
            predicted_labels = None
            if multi_class:
                predicted_labels = output.argmax(dim=-1).squeeze()
            else:
                predicted_labels = (output > 0.5).int().squeeze()

            for i in range(num_groups):
                group_mask = (groups == i)
                num_elems_for_group[run][i] += group_mask.sum().item()
                group_acc = (predicted_labels[group_mask] == ground_truth_labels[group_mask]).sum().item()
                acc_per_group[run][i] += group_acc

    # the number of elements per group does not need to be calculated for each run, but it makes writing this a bit easier
    acc_per_group = torch.tensor(acc_per_group) / torch.tensor(num_elems_for_group)

    group_acc_averaged_over_runs = acc_per_group.mean(dim=0)
    worst_group_acc = group_acc_averaged_over_runs.min().item()
    worst_group = group_acc_averaged_over_runs.argmin().item()
    macro_avg_group_acc = group_acc_averaged_over_runs.mean().item()
    std_dev_group_acc = group_acc_averaged_over_runs.std().item()
    std_dev_per_group = acc_per_group.std(dim=0)
    std_dev_for_worst_group = std_dev_per_group[worst_group].float().item()
    if not suppress_log:
        print(f"Macro average (over restarts) group accuracy = {macro_avg_group_acc:.4g}")
        print(f"Min group accuracy = {worst_group_acc:.4g}, group idx = {worst_group}")
        print(f"Group accuracies averaged over run = {group_acc_averaged_over_runs}")

    return round(macro_avg_group_acc, 5), round(worst_group_acc, 5), worst_group, round(std_dev_group_acc, 5), round(std_dev_for_worst_group, 5)

def get_restart_macro_avg_acc_over_labels_with_stddev(
    dl_test: torch.utils.data.DataLoader, model_run_dir: str, model: torch.nn.Sequential, device: str, num_classes: int,
    multi_class: bool = False, suppress_log: bool = False) -> tuple[float, float]:
    restarts = len(os.listdir(model_run_dir))
    acc_per_label, num_elems_per_label = np.zeros((restarts, num_classes)), np.zeros((restarts, num_classes))
    model.eval()
    for run in range(restarts):
        run_file = os.path.join(model_run_dir, f"run_{run}.pt")
        model.load_state_dict(torch.load(run_file))
        model = model.to(device)
        model.eval()
        for data, ground_truth_labels, _, _ in dl_test:
            data, ground_truth_labels = data.to(device), ground_truth_labels.to(device)
            output = model(data).squeeze()
            predicted_labels = None
            if multi_class:
                predicted_labels = output.argmax(dim=-1).squeeze()
            else:
                predicted_labels = (output > 0.5).int().squeeze()

            for i in range(num_classes):
                label_mask = ground_truth_labels == i
                acc_per_label[run][i] += (predicted_labels[label_mask] == ground_truth_labels[label_mask]).sum().item()
                num_elems_per_label[run][i] += label_mask.sum().item()

    acc_per_label = torch.tensor(acc_per_label) / torch.tensor(num_elems_per_label)
    macro_avg_per_run = acc_per_label.mean(dim=1)
    macro_avg_over_labels, stddev_over_labels = macro_avg_per_run.mean().item(), macro_avg_per_run.std(dim=0).item()
    if not suppress_log:
        print("---- Macro averaged over labels accuracy ----")
        print(f"Macro average and stdev over labels = {macro_avg_over_labels:.4g}, {stddev_over_labels:.4g}")

    return round(macro_avg_over_labels, 5), round(stddev_over_labels, 5)

def get_avg_acc_with_stddev(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, model_dir: str,
    num_classes: int) -> tuple[float, float]:
    num_runs = len(os.listdir(model_dir))
    acc_sum = np.zeros(num_runs)
    with torch.no_grad():
        for run_idx in range(num_runs):
            model.load_state_dict(torch.load(f"{model_dir}/run_{run_idx}.pt"))
            model = model.to(device)
            acc_sum[run_idx] = test_model_accuracy(model, test_dloader, device, multi_class=num_classes > 2, suppress_log=True)

    return acc_sum.mean(), acc_sum.std()

def get_avg_rob_metrics(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, model_dir: str,
                  eps: float, loss_fn: str, has_conv: bool) -> tuple[float, float, float, float]:
    num_runs = len(os.listdir(model_dir))
    deltas, ls, us = np.zeros(num_runs), np.zeros(num_runs), np.zeros(num_runs)
    with torch.no_grad():
        for run_idx in range(num_runs):
            model.load_state_dict(torch.load(f"{model_dir}/run_{run_idx}.pt"))
            model = model.to(device)
            _, d, l, u = test_delta_input_robustness(test_dloader, model, eps, 0, loss_fn, device, has_conv, suppress_log=True)
            deltas[run_idx] = d
            ls[run_idx] = l
            us[run_idx] = u
    return deltas.mean(), ls.mean(), us.mean(), deltas.std(), ls.std(), us.std()

# ----------------------------------------------------------------------------------------
# ------------------------------------ Test functions ------------------------------------
# ----------------------------------------------------------------------------------------
def test_avg_delta(dset_name: str):
    assert dset_name in ["isic", "plant", "decoy_mnist", "imagenet"]
    dl_test, model_dir, model, device, loss_fn, eps, has_conv = None, None, None, "cuda:0", None, None, True
    if dset_name == "isic":
        data_root = "/vol/bitbucket/mg2720/isic/"

        isic_test_grouped = isic.ISICDataset(data_root, is_train=False, grouped=True)
        dl_test = isic.get_loader_from_dataset(isic_test_grouped, batch_size=256, shuffle=False)
        model = LesionNet(3, 1)
        loss_fn = "binary_cross_entropy"
        model_dir = "saved_experiment_models/performance/isic"
        eps = 1
    elif dset_name == "plant":
        split_root = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
        data_root = "/vol/bitbucket/mg2720/plant/rgb_data"
        masks_file = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
        _ = plant.PlantDataset(split_root, data_root, masks_file, 2, True)
        plant_test_2 = plant.PlantDataset(split_root, data_root, masks_file, 2, False)
        dl_test = plant.get_dataloader(plant_test_2, 10)
        model = PlantNet(3, 1)
        loss_fn = "binary_cross_entropy"
        model_dir = "saved_experiment_models/performance/plant"
        eps = 0.01
    else :
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        loss_fn = "cross_entropy"
        model_dir = "saved_experiment_models/performance/decoy_mnist"
        has_conv = False
        eps = 0.1
    for method in ["ibp_ex+r3"]:
        avg_delta = get_avg_rob_metrics(model, dl_test, device, model_dir + f"/{method}", eps, loss_fn, has_conv)
        print(f"Method {method} avg delta = {avg_delta}")

def test_macro_over_labels_and_wg_acc(dset_name: str):
    assert dset_name in ["isic", "plant", "decoy_mnist", "imagenet"]
    dev = torch.device("cuda:1")
    dl_test, model, num_classes, num_groups = None, None, None, None
    if dset_name == "isic":
        data_root = "/vol/bitbucket/mg2720/isic"
        test_dset = isic.ISICDataset(data_root, is_train=False, grouped=True)
        dl_test = isic.get_loader_from_dataset(test_dset, batch_size=256, shuffle=False)
        model = LesionNet(3, 1)
        num_classes = 2
        num_groups = 3
    elif dset_name == "plant":
        split_root = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
        data_root = "/vol/bitbucket/mg2720/plant/rgb_data"
        masks_file = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
        plant_test_2 = plant.PlantDataset(split_root, data_root, masks_file, 2, False)
        dl_test = plant.get_dataloader(plant_test_2, 25)
        model = PlantNet(3, 1)
        num_classes = 2
    elif dset_name == "imagenet":
        num_classes = 6
        model = SalientImageNet()
        data_dir = "/vol/bitbucket/mg2720/val.X"
        masks_dir = "/vol/bitbucket/mg2720/salient_imagenet_dataset"

        test_imgnet = salient_imagenet.LazyImageNetTestDataset(data_dir, masks_dir, preprocess=(ResNet18_Weights.DEFAULT.transforms()))
        dl_test = salient_imagenet.get_dataloader(test_imgnet, 100)
    else: # decoy_mnist
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        num_classes = 10
    for method in ["std"]:#, "r3", "r4", "ibp_ex", "ibp_ex+r3", "smooth_r3", "pgd_r4", "rand_r4"]:
        print(f"Method {method}")
        model_dir = f"saved_experiment_models/performance/{dset_name}/{method}"
        macro_avg_over_labels, _ = get_restart_macro_avg_acc_over_labels_with_stddev(
            dl_test, model_dir, model, dev, num_classes, multi_class=num_classes > 2
        )
        macro_avg_group_acc, worst_group_acc, worst_group, _, _ = get_restart_avg_and_worst_group_accuracy_with_stddev(
            dl_test, model_dir, model, dev, num_groups, multi_class=num_classes > 2
        )
        print(f"Macro average over labels = {macro_avg_over_labels}")
        print(f"Macro average group acc = {macro_avg_group_acc}, worst group acc = {worst_group_acc}, worst group = {worst_group}")

def lambda_rcs_acc(net, test_dl, device):
    corrs, noisy_corrs, dset_size = 0, 0, 0
    for batch_x, batch_y, batch_mask, _ in test_dl:
        noise = torch.randn(size=batch_x.shape) * 2
        batch_x, batch_y, batch_mask, noise = batch_x.to(device), batch_y.to(device), batch_mask.to(device), noise.to(device)
        with torch.no_grad():
            pred_y = torch.argmax(net(batch_x), dim=-1)
            noisy_preds = torch.argmax(net(batch_x + noise * batch_mask), dim=-1)
            corrs += (pred_y == batch_y).sum()
            noisy_corrs += (noisy_preds == batch_y).sum()
            dset_size += len(pred_y)
    acc, noisy_acc = corrs / dset_size, noisy_corrs / dset_size
    return acc, noisy_acc

def get_avg_rcs(dl_test_spurious: DataLoader, dl_test_core: DataLoader, model_run_dir: str, device: str, suppress_logs: bool = False) -> float:
    model = SalientImageNet()
    model.eval()
    avg_rcs, avg_spur_acc = 0, 0

    for run_file in os.listdir(model_run_dir):
        model.load_state_dict(torch.load(f"{model_run_dir}/{run_file}"))
        model = model.to(device)
        acc_spur, noisy_acc_spur = lambda_rcs_acc(model, dl_test_spurious, device)
        acc_core, noisy_acc_core = lambda_rcs_acc(model, dl_test_core, device)

        avg_acc = (noisy_acc_core + noisy_acc_spur) / 2
        rcs_for_run = (noisy_acc_core - noisy_acc_spur) / (2 * min(avg_acc, 1 - avg_acc))
        avg_rcs += round(float(rcs_for_run), 5)
        avg_spur_acc += round(float(acc_spur), 5)

        #* We make a slightly different calculation. Core and spurious accuracies are obtained by perturbing core and spurious masks
        #* instead of perturbing 1-core and 1-spurious masks as proposed in the original paper.
        #* Approximating the obtained core_acc as 1-core_acc (that we would have gotten if we had perturbed 1-core_mask),
        #* similarly for spu_acc, we report RCS as spu_acc-core_acc/denominator.
        if not suppress_logs:
            print("RCS:", rcs_for_run)
            print(f"Core accuracy = {acc_core}, spurious accuracy = {acc_spur}")
    avg_rcs /= len(os.listdir(model_run_dir))
    avg_spur_acc /= len(os.listdir(model_run_dir))
    if not suppress_logs:
        print(f"Average RCS = {-avg_rcs}")
        print(f"Average spurious accuracy = {avg_spur_acc}")

    return -avg_rcs, avg_spur_acc

def get_rcs_for_run(dl_test_spurious: DataLoader, dl_test_core: DataLoader, model: SalientImageNet, device: str, suppress_logs: bool = False) -> float:
    model.eval()
    model = model.to(device)
    acc_spur, noisy_acc_spur = lambda_rcs_acc(model, dl_test_spurious, device)
    acc_core, noisy_acc_core = lambda_rcs_acc(model, dl_test_core, device)

    avg_acc = (noisy_acc_core + noisy_acc_spur) / 2
    rcs_for_run = (noisy_acc_core - noisy_acc_spur) / (2 * min(avg_acc, 1 - avg_acc))
    rcs_for_run = round(float(rcs_for_run), 5)
    acc_spur = round(float(acc_spur), 5)
    if not suppress_logs:
        print(f"Avg Acc = {avg_acc}, 2 * min(avg_acc, 1 - avg_acc) = {2 * min(avg_acc, 1 - avg_acc)}")
        print("RCS:", -rcs_for_run)
        print(f"Core accuracy = {acc_core}, spurious accuracy = {acc_spur}")
        print(f"Noisy core accuracy = {noisy_acc_core}, noisy spurious accuracy = {noisy_acc_spur}")

    # Report (-RCS)
    return  -rcs_for_run, acc_spur
