import os
import torch
from torch.utils.data import DataLoader
import sys
from datasets import derma_mnist, plant, decoy_mnist
from models.R4_models import DermaNet, PlantNet
from models.fully_connected import FCNAugmented
from models.pipeline import test_delta_input_robustness

def worst_group_acc(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, num_classes: int,
                    runs_dir_root: str, suppress_log=False) -> tuple[float, int]:
    multi_class = num_classes > 2
    acc_per_class = [0] * num_classes
    elems_per_class = [0] * num_classes
    num_runs = len(os.listdir(runs_dir_root))
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
                        correct = (output.argmax(dim=0) == test_label).sum().item()
                    else:
                        correct = ((output > 0.5) == (test_label)).sum().item()
                    acc_per_class[int(test_label)] += correct / num_runs
                    elems_per_class[int(test_label)] += 1 / num_runs
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

def get_avg_delta(model: torch.nn.Sequential, test_dloader: DataLoader, device: str, model_dir: str,
                  eps: float, loss_fn: str, has_conv: bool) -> float:
    delta_sum = 0
    num_runs = len(os.listdir(model_dir))
    with torch.no_grad():
        for run_idx in range(num_runs):
            model.load_state_dict(torch.load(f"{model_dir}/run_{run_idx}.pt"))
            model = model.to(device)
            delta_sum += test_delta_input_robustness(test_dloader, model, eps, 0, loss_fn, device, has_conv)[2]
    return delta_sum / num_runs

def test_avg_delta(dset_name: str):
    assert dset_name in ["derma_mnist", "plant", "decoy_mnist"]
    dl_test, model_dir, model, device, loss_fn, eps, has_conv = None, None, None, "cuda:0", None, None, True
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
    for method in ["std", "r3", "r4", "ibp_ex", "ibp_ex+r3"]:
        avg_delta = get_avg_delta(model, dl_test, device, model_dir + f"/{method}", eps, loss_fn, has_conv)
        print(f"Method {method} avg delta = {avg_delta:.4g}")


def test_worst_group_acc():
    assert len(sys.argv) == 2
    assert sys.argv[1] in ["derma_mnist", "plant", "decoy_mnist"]
    dev = torch.device("cuda:0")
    if sys.argv[1] == "derma_mnist":
        IMG_SIZE = 64
        test_dset = derma_mnist.DecoyDermaMNIST(False, size=IMG_SIZE)
        dl_test = derma_mnist.get_dataloader(test_dset, 256)
        model = DermaNet(3, IMG_SIZE, 1)
        for method in ["std", "r3", "r4", "ibp_ex", "ibp_ex+r3"]:
            worst_group_acc(model, dl_test, dev, 2, f"saved_experiment_models/performance/derma_mnist/{method}")
    elif sys.argv[1] == "plant":
        SPLIT_ROOT = "/vol/bitbucket/mg2720/plant/rgb_dataset_splits"
        DATA_ROOT = "/vol/bitbucket/mg2720/plant/rgb_data"
        MASKS_FILE = "/vol/bitbucket/mg2720/plant/mask/preprocessed_masks.pyu"
        plant_test_2 = plant.PlantDataset(SPLIT_ROOT, DATA_ROOT, MASKS_FILE, 2, False)
        dl_test = plant.get_dataloader(plant_test_2, 50)
        model = PlantNet(3, 1)
        for method in ["std", "r3", "r4", "ibp_ex", "ibp_ex+r3"]:
            print(f"Method {method}")
            worst_group_acc(model, dl_test, dev, 2, f"saved_experiment_models/performance/plant/{method}")
    elif sys.argv[1] == "decoy_mnist":
        dl_train_no_mask, dl_test_no_mask = decoy_mnist.get_dataloaders(1000, 1000)
        _, dl_test = decoy_mnist.get_masked_dataloaders(dl_train_no_mask, dl_test_no_mask)
        model = FCNAugmented(784, 10, 512, 1)
        for method in ["std", "r3", "r4", "ibp_ex", "ibp_ex+r3"]:
            worst_group_acc(model, dl_test, dev, 10, f"saved_experiment_models/performance/decoy_mnist/{method}")
    else:
        raise ValueError("Only 'derma_mnist' and 'plant' are supported")
