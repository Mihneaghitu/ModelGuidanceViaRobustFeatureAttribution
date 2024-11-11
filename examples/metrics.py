import os
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from datasets import derma_mnist, plant, decoy_mnist, salient_imagenet
from models.R4_models import DermaNet, PlantNet, SalientImageNet
from models.fully_connected import FCNAugmented
from models.pipeline import test_delta_input_robustness, test_model_accuracy
import numpy as np


def worst_group_acc_no_load(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, num_classes: int, suppress_log=False) -> tuple[float, int]:
    multi_class = num_classes > 2
    acc_per_class = [0] * num_classes
    elems_per_class = [0] * num_classes
    with torch.no_grad():
        model = model.to(device)
        for test_batch, test_labels, _ in test_dloader:
            for i in range(test_batch.shape[0]):
                test_point, test_label = test_batch[i].to(device), test_labels[i].to(device)
                # because it is a single point, we need to add a batch dimension
                test_point = test_point.unsqueeze(0)
                output = model(test_point).squeeze()
                correct = 0
                if multi_class:
                    correct = (output.argmax(dim=0) == test_label).item()
                else:
                    correct = ((output > 0.5) == (test_label)).item()
                acc_per_class[int(test_label)] += correct
                elems_per_class[int(test_label)] += 1
    min_acc, min_class = 1, -1
    for i in range(num_classes):
        if elems_per_class[i] > 0:
            acc_per_class[i] /= elems_per_class[i]
            if acc_per_class[i] < min_acc:
                min_acc = acc_per_class[i]
                min_class = i
    if not suppress_log:
        print(f"Worst class accuracy = {min_acc:.4g} for class {min_class}")

    return round(min_acc, 3), min_class

def worst_group_acc(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, num_classes: int,
                    runs_dir_root: str, suppress_log=False) -> tuple[float, int]:
    multi_class = num_classes > 2
    acc_per_class = [0] * num_classes
    elems_per_class = [0] * num_classes
    num_runs = len(os.listdir(runs_dir_root))
    model.eval()
    with torch.no_grad():
        for run_idx in range(num_runs):
            model.load_state_dict(torch.load(f"{runs_dir_root}/run_{run_idx}.pt"))
            model = model.to(device)
            for test_batch, test_labels, _ in test_dloader:
                for i in range(test_batch.shape[0]):
                    test_point, test_label = test_batch[i].to(device), test_labels[i].to(device)
                    # because it is a single point, we need to add a batch dimension
                    test_point = test_point.unsqueeze(0)
                    output = model(test_point).squeeze()
                    correct = 0
                    if multi_class:
                        correct = (output.argmax(dim=0) == test_label).item()
                    else:
                        correct = ((output > 0.5) == (test_label)).item()
                    acc_per_class[int(test_label)] += correct
                    elems_per_class[int(test_label)] += 1
    min_acc, min_class = 1, -1
    for i in range(num_classes):
        if elems_per_class[i] > 0:
            acc_per_class[i] /= elems_per_class[i]
            if acc_per_class[i] < min_acc:
                min_acc = acc_per_class[i]
                min_class = i
    if not suppress_log:
        print(f"Worst class accuracy = {min_acc:.4g} for class {min_class}")

    return round(min_acc, 3), min_class

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

def test_avg_delta(dset_name: str):
    assert dset_name in ["derma_mnist", "plant", "decoy_mnist"]
    dl_test, model_dir, model, device, loss_fn, eps, has_conv = None, None, None, "cuda:1", None, None, True
    if dset_name == "derma_mnist":
        test_dset = derma_mnist.DecoyDermaMNIST(False, size=64)
        dl_test = derma_mnist.get_dataloader(test_dset, 256)
        model = DermaNet(3, 64, 1)
        loss_fn = "binary_cross_entropy"
        model_dir = "saved_experiment_models/performance/derma_mnist"
        eps = 0.05
    elif dset_name == "plant":
        split_root = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
        data_root = "/vol/bitbucket/mg2720/plant/rgb_data"
        masks_file = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
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
    for method in ["r4_pmo", "rand_r4_pmo", "pgd_r4_pmo"]:#["std", "r3", "r4", "ibp_ex", "ibp_ex+r3", "smooth_r3", "pgd_r4", "rand_r4"]:
        avg_delta = get_avg_rob_metrics(model, dl_test, device, model_dir + f"/{method}", eps, loss_fn, has_conv)
        print(f"Method {method} avg delta = {avg_delta}")


def test_worst_group_acc(dset_name: str):
    assert dset_name in ["derma_mnist", "plant", "decoy_mnist", "imagenet"]
    dev = torch.device("cuda:0")
    img_size, dl_test, model, num_classes = None, None, None, None
    if dset_name == "derma_mnist":
        img_size = 64
        test_dset = derma_mnist.DecoyDermaMNIST(False, size=img_size)
        dl_test = derma_mnist.get_dataloader(test_dset, 256)
        model = DermaNet(3, img_size, 1)
        num_classes = 2
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

        test_imgnet = salient_imagenet.LazyImageNetTestDataset(data_dir, masks_dir, False)
        dl_test = salient_imagenet.get_dataloader(test_imgnet, 100)
    else: # decoy_mnist
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        num_classes = 10

    for method in ["std"]:#, "r3", "r4", "ibp_ex", "ibp_ex+r3", "smooth_r3", "pgd_r4", "rand_r4"]:
        print(f"Method {method}")
        worst_group_acc(model, dl_test, dev, num_classes, f"saved_experiment_models/performance/{dset_name}/{method}")


def test_avg_acc(dset_name: str):
    assert dset_name in ["derma_mnist", "plant", "decoy_mnist", "imagenet"]
    dev = torch.device("cuda:0")
    img_size, dl_test, model, num_classes = None, None, None, None
    if dset_name == "derma_mnist":
        img_size = 64
        test_dset = derma_mnist.DecoyDermaMNIST(False, size=img_size)
        dl_test = derma_mnist.get_dataloader(test_dset, 256)
        model = DermaNet(3, img_size, 1)
        num_classes = 2
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

        test_imgnet = salient_imagenet.LazyImageNetTestDataset(data_dir, masks_dir, False)
        dl_test = salient_imagenet.get_dataloader(test_imgnet, 100)
    else: # decoy_mnist
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        num_classes = 10

    for method in ["std"]:#, "r3", "r4", "ibp_ex", "ibp_ex+r3", "smooth_r3", "pgd_r4", "rand_r4"]:
        print(f"Method {method}")
        model_dir = f"saved_experiment_models/performance/{dset_name}/{method}"
        mean_acc, std_acc = get_avg_acc_with_stddev(model, dl_test, dev, model_dir, num_classes)
        print(f"Method {method} avg acc = {mean_acc} with std dev {std_acc}")
