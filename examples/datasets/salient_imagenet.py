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
    "n01818515": 0, # Macaw
    "n02007558": 1, # Flamingo
    "n01770393": 2, # Scorpion
    "n01749939": 3, # Green Mamba
    "n01944390": 4, # Snail
    "n01698640": 5, # American Alligator
}

class ImageNetDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, is_train: bool, train_proportion:float = 0.75, split_seed: int = 0):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.nn.functional.one_hot(torch.tensor([label_idx]), num_classes=6).float()

        self.data_tensors = []
        self.label_tensors = []
        self.mask_tensors = []
        for wnid in self.relevant_classes[:1]:
            data_subdir_path = os.path.join(data_dir, wnid)
            for fname in os.listdir(data_subdir_path):
                data_img = Image.open(os.path.join(data_subdir_path, fname))
                data_img = data_transform(data_img)
                self.data_tensors.append(data_img)
                assert self.data_tensors == [] or self.data_tensors[-1].shape == data_img.shape
                # Now search for the masks
                masks_subdir = os.path.join(masks_dir, wnid)
                masks_for_input = []
                for feature_dir in os.listdir(masks_subdir):
                    if not feature_dir.endswith(".csv"): # just ignore the map
                        feature_dir_path = os.path.join(masks_subdir, feature_dir)
                        # Need to split because of the extension
                        if fname.split(".")[0] in os.listdir(feature_dir_path):
                            print(f"Found mask for {fname} in {feature_dir_path}")
                            mask = Image.open(os.path.join(feature_dir_path, fname))
                            mask = data_transform(mask)
                            masks_for_input.append(mask)
                self.mask_tensors.append(masks_for_input)
                self.label_tensors.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        self.data_tensors = np.array(self.data_tensors)
        self.label_tensors = np.array(self.label_tensors)
        self.mask_tensors = np.array(self.mask_tensors)
        all_split_indices = np.random.permutation(len(self.data_tensors))
        num_train = int(train_proportion * len(self.data_tensors))
        split_indices = all_split_indices[:num_train] if is_train else all_split_indices[num_train:]

        self.data_tensors = torch.tensor(self.data_tensors[split_indices])
        self.label_tensors = torch.tensor(self.label_tensors[split_indices])
        self.mask_tensors = torch.tensor(self.mask_tensors[split_indices])

    def __len__(self):
        return self.data_tensors.size(0)

    def __getitem__(self, idx):
        return self.data_tensors[idx], self.label_tensors[idx], self.mask_tensors[idx]

def get_dataloader(plant_dset: ImageNetDataset, batch_size: int):
    return DataLoader(plant_dset, batch_size=batch_size, shuffle=False)
