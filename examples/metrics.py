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

def dilate_erode(masks, dilate=True, iterations=15, kernel=5):
    ''' Dilate or erode tensor of soft segmentation masks'''
    assert kernel % 2 == 1
    half_k = kernel // 2
    batch_size, _, side_len, _ = masks.shape

    out = masks[:,0,:,:].clone()
    padded = torch.zeros(batch_size, side_len+2*half_k, side_len+2*half_k, device=masks.device)
    if not dilate:
        padded = 1 + padded
    for itr in range(iterations):
        all_padded = []
        centered = padded.clone()
        centered[:, half_k:half_k+side_len, half_k:half_k+side_len]; all_padded.append(centered)
        for j in range(1, half_k+1):
            left, right, up, down = [padded.clone() for _ in range(4)]
            left[:, half_k-j:half_k-j+side_len, half_k:half_k+side_len] = out; all_padded.append(left)
            right[:, half_k+j:half_k+j+side_len, half_k:half_k+side_len] = out; all_padded.append(right)
            up[:, half_k:half_k+side_len, half_k+j:half_k+j+side_len] = out; all_padded.append(up)
            down[:, half_k:half_k+side_len, half_k-j:half_k-j+side_len] = out; all_padded.append(down)
        all_padded = torch.stack(all_padded)
        out = torch.max(all_padded, dim=0)[0] if dilate else torch.min(all_padded, dim=0)[0]
        out = out[:, half_k:half_k+side_len, half_k:half_k+side_len]

    out = torch.stack([out, out, out], dim=1)
    out = out / torch.max(out)
    return out


def rel_score(core_acc, spur_acc):
    '''
    Computes relative core sensitivity for scalar values core_acc and spur_acc
    '''
    avg = 0.5 * (core_acc + spur_acc)
    if avg == 0 or avg == 1:
        return 0

    return (core_acc - spur_acc) / (2 * min(avg, 1-avg))

def core_spur_accuracy(test_dloader, model, device, noise_sigma=0.25, num_trials=10, apply_norm=False):
    '''
    Core regions are taken to be dilated core masks, and spurious regions are 1-dilated core masks
    Use Salient Imagenet test set for 'dset', or any dataset with soft segmentation masks for core regions.
    Returns overall core and spurious accuracy, as well as per class metrics.
    '''
    normalize = ResNet18_Weights.DEFAULT.transforms()
    cnt_by_class = dict({i:0 for i in range(1000)})
    core_cc_by_class, spur_cc_by_class = dict({i:0 for i in range(1000)}), dict({i:0 for i in range(1000)})

    for dat in test_dloader:
        imgs, labels, masks, _ = [x.cuda(device) for x in dat]

        idx_with_masks = (masks.flatten(1).sum(1) != 0)
        imgs, labels, masks = [x[idx_with_masks] for x in [imgs, labels, masks]]

        masks = dilate_erode(masks)

        for _ in range(num_trials):
            noise = torch.randn_like(imgs, device=imgs.device) * noise_sigma
            noisy_core, noisy_spur = [torch.clamp(imgs + (x * noise), 0, 1) for x in [masks, 1-masks]]
            if apply_norm:
                noisy_core, noisy_spur = [normalize(x) for x in [noisy_core, noisy_spur]]
            noisy_core_preds, noisy_spur_preds = [model(x).argmax(1) for x in [noisy_core, noisy_spur]]

            for y in np.unique(labels.cpu().numpy()):
                core_cc_by_class[y] += (noisy_spur_preds[labels == y] == y).sum().item()
                spur_cc_by_class[y] += (noisy_core_preds[labels == y] == y).sum().item()
                cnt_by_class[y] += (labels == y).sum().item()

    total_cnt, total_core_cc, total_spur_cc = 0, 0, 0
    core_acc_by_class, spur_acc_by_class = {}, {}
    for c in cnt_by_class:
        if cnt_by_class[c] == 0:
            continue
        total_core_cc += core_cc_by_class[c]
        total_spur_cc += spur_cc_by_class[c]
        total_cnt += cnt_by_class[c]
        core_acc_by_class[c] = core_cc_by_class[c] / cnt_by_class[c]
        spur_acc_by_class[c] = spur_cc_by_class[c] / cnt_by_class[c]

    core_acc, spur_acc = [np.average(list(x.values())) for x in [core_acc_by_class, spur_acc_by_class]]
    return core_acc, spur_acc, core_acc_by_class, spur_acc_by_class


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

def get_rcs(dl_test: DataLoader, model_run_dir: str, device: str, eps: str, suppress_log: bool = False) -> float:
    model = SalientImageNet()
    model.eval()
    avg_rcs = 0
    for run_file in os.listdir(model_run_dir):
        model.load_state_dict(torch.load(f"{model_run_dir}/{run_file}"))
        model = model.to(device)
        core_acc, spur_acc, _, _ = core_spur_accuracy(dl_test, model, device, noise_sigma=eps, apply_norm=False)
        rcs = rel_score(core_acc, spur_acc)
        avg_rcs += rcs
        if not suppress_log:
            print(f"Core acc = {core_acc}, spur acc = {spur_acc}")
            print(f"Rcs = {rcs}")

    avg_rcs /= len(os.listdir(model_run_dir))

    return avg_rcs