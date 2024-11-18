
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from skimage.segmentation import slic
import cv2

class ISICDataset(Dataset):
    def __init__(self, data_root: str, is_train: bool = True, grouped: bool = False):
        self.data_root_cancer = os.path.join(data_root, "processed/cancer")
        self.data_root_no_cancer = os.path.join(data_root, "processed/no_cancer")
        self.data_root_patch_no_cancer = os.path.join(data_root, "processed/patch_no_cancer_again")
        self.masks_root = os.path.join(data_root, "segmentation")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
        ])
        self.data_paths, self.labels, self.masks, self.groups = [], [], [], []
        self.__collect_path_from_dir(self.data_root_no_cancer)
        self.__collect_path_from_dir(self.data_root_cancer)
        self.__collect_path_from_dir(self.data_root_patch_no_cancer)

        all_indices = np.random.permutation(len(self.data_paths))
        threshold = int(len(all_indices) * 0.8)
        split_indices = all_indices[:threshold] if is_train else all_indices[threshold:]
        self.data_paths = [self.data_paths[i] for i in split_indices]
        self.labels = torch.tensor([self.labels[i] for i in split_indices]).float()
        self.masks = [self.masks[i] for i in split_indices]
        self.groups = [self.groups[i] for i in split_indices]
        self.grouped = grouped

    def __collect_path_from_dir(self, dir_path: str) -> None:
        with_cancer = dir_path == self.data_root_cancer
        with_patch = dir_path == self.data_root_patch_no_cancer

        _fnames = sorted(os.listdir(dir_path))
        data_paths, labels, masks, groups = [], None, [], []

        if with_cancer:
            labels = [1] * len(_fnames)
        else:
            labels = [0] * len(_fnames)

        for fname in _fnames:
            data_paths.append(os.path.join(dir_path, fname))
            # if it does not have a patch, the mask is 0, and its group is 2 (patch_no_cancer)
            if with_patch:
                masks.append(os.path.join(self.masks_root, fname))
                groups.append(2)
            # if it has a patch, the mask is 1
            else:
                masks.append("-1")
                # if it has cancer, the group is 0, otherwise 1
                if with_cancer:
                    groups.append(0)
                else:
                    groups.append(1)

        self.data_paths += data_paths
        self.labels += labels
        self.masks += masks
        self.groups += groups

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx) -> tuple:
        img = Image.open(self.data_paths[idx])
        img = self.transform(img).float()
        label = self.labels[idx]
        group = self.groups[idx]
        mask = None
        if self.masks[idx] != "-1":
            mask = Image.open(self.masks[idx])
            mask = self.transform(mask)
        else:
            mask = torch.zeros_like(img)
        mask = mask.float()

        if self.grouped:
            return img, label, mask, group
        else:
            return img, label, mask


def get_loader_from_dataset(dataset: ISICDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
