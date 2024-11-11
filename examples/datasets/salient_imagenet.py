import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import pickle
import pandas as pd
import copy
import urllib.request

WNID_TO_LABEL_DICT = {
    "n01818515": 0, # 88, Macaw
    "n02007558": 1, # 130, Flamingo
    "n01770393": 2, # 71,Scorpion
    "n01749939": 3, # 64, Green Mamba
    "n01944390": 4, # 113, Snail
    "n01698640": 5  # 50, American Alligator
}

class LazyImageNetTrainDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, is_train: bool, preprocess: callable = None, train_proportion: float = 0.75, split_seed: int = 0):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # As per https://pytorch.org/hub/pytorch_vision_resnet/
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_transform = preprocess if preprocess else self.data_transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.tensor([label_idx])

        data_paths, mask_paths, labels = [], [], []
        self.data_paths, self.mask_paths, self.labels = [], [], []
        for wnid in self.relevant_classes:
            data_subdir_path = os.path.join(data_dir, wnid)
            for fname in os.listdir(data_subdir_path):
                curr_data_path = os.path.join(data_subdir_path, fname)
                # Now search for the masks
                masks_subdir = os.path.join(masks_dir, wnid)
                # read the csv
                mmaps = pd.read_csv(os.path.join(masks_subdir, "image_names_map.csv"))
                cnt = 0
                for c in mmaps.columns:
                    # get the column as a list of strings
                    mask_names = mmaps[c].tolist()
                    if fname.split(".")[0] in mask_names:
                        curr_mask_path = os.path.join(masks_subdir, f"feature_{c}", fname)
                        data_paths.append(curr_data_path)
                        mask_paths.append(curr_mask_path)
                        labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
                        cnt += 1
                if cnt == 0:
                    data_paths.append(curr_data_path)
                    mask_paths.append("-1") # dummy flag
                    labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
            # if is_train:
            #     self.data_paths += data_paths[:1000]
            #     self.labels += labels[:1000]
            #     self.mask_paths += mask_paths[:1000]
            # else:
            #     self.data_paths += data_paths[1000:]
            #     self.labels += labels[1000:]
            #     self.mask_paths += mask_paths[1000:]
            # data_paths, mask_paths, labels = [], [], []

        # all_split_indices = np.random.permutation(len(data_paths))
        # num_train = int(train_proportion * len(data_paths))
        # split_indices = all_split_indices[:num_train] if is_train else all_split_indices[num_train:]
        split_indices = list(range(len(data_paths)))

        self.data_paths = [data_paths[i] for i in split_indices]
        self.label_tensors = torch.stack(labels).squeeze()
        self.mask_paths = [mask_paths[i] for i in split_indices]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_img = Image.open(self.data_paths[idx]).convert("RGB")
        data_tensor = self.data_transform(data_img)
        mask_tensor = torch.zeros(1, 224, 224, dtype=torch.float32) / 100
        if not self.mask_paths[idx] == "-1":
            mask_img = Image.open(self.mask_paths[idx])
            mask_img = self.mask_transform(mask_img)
            mask_tensor = 1 - mask_img
            assert mask_tensor.shape == (1, 224, 224)
        mask_tensor = mask_tensor.repeat(3, 1, 1)
        assert mask_tensor.shape == (3, 224, 224), data_tensor.shape == (3, 224, 224)
        return data_tensor, self.label_tensors[idx], mask_tensor


class LazyImageNetTestDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, is_train: bool, preprocess: callable = None, train_proportion: float = 0.75, split_seed: int = 0):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # As per https://pytorch.org/hub/pytorch_vision_resnet/
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_transform = preprocess if preprocess else self.data_transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.tensor([label_idx])

        data_paths, mask_paths, labels = [], [], []
        self.data_paths, self.mask_paths, self.labels = [], [], []
        for wnid in self.relevant_classes:
            data_subdir_path = os.path.join(data_dir, wnid)
            for fname in os.listdir(data_subdir_path):
                curr_data_path = os.path.join(data_subdir_path, fname)
                # Now search for the masks
                masks_subdir = os.path.join(masks_dir, wnid)
                # read the csv
                mmaps = pd.read_csv(os.path.join(masks_subdir, "image_names_map.csv"))
                cnt = 0
                for c in mmaps.columns:
                    # get the column as a list of strings
                    mask_names = mmaps[c].tolist()
                    if fname.split(".")[0] in mask_names:
                        curr_mask_path = os.path.join(masks_subdir, f"feature_{c}", fname)
                        data_paths.append(curr_data_path)
                        mask_paths.append(curr_mask_path)
                        labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
                        cnt += 1
                if cnt == 0:
                    data_paths.append(curr_data_path)
                    mask_paths.append("-1") # dummy flag
                    labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        self.data_paths = data_paths
        self.label_tensors = torch.stack(labels).squeeze()
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_img = Image.open(self.data_paths[idx]).convert("RGB")
        data_tensor = self.data_transform(data_img)
        mask_tensor = torch.zeros(1, 224, 224, dtype=torch.float32) / 100
        if not self.mask_paths[idx] == "-1":
            mask_img = Image.open(self.mask_paths[idx])
            mask_img = self.mask_transform(mask_img)
            mask_tensor = 1 - mask_img
            assert mask_tensor.shape == (1, 224, 224)
        mask_tensor = mask_tensor.repeat(3, 1, 1)
        assert mask_tensor.shape == (3, 224, 224), data_tensor.shape == (3, 224, 224)
        return data_tensor, self.label_tensors[idx], mask_tensor

def get_dataloader(imgnet_dset: Dataset, batch_size: int, drop_last: bool = False):
    return DataLoader(imgnet_dset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
