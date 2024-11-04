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

        init_data_tensors = []
        init_label_tensors = []
        init_mask_tensors = []
        for wnid in self.relevant_classes:
            data_subdir_path = os.path.join(data_dir, wnid)
            for fname in os.listdir(data_subdir_path):
                data_img = Image.open(os.path.join(data_subdir_path, fname))
                data_img = data_transform(data_img)
                # for some reason, some images have only 1 channel, just discard them
                if init_data_tensors and data_img.shape != init_data_tensors[-1].shape:
                    continue
                assert not init_data_tensors or init_data_tensors[-1].shape == data_img.shape
                init_data_tensors.append(data_img)
                # Now search for the masks
                masks_subdir = os.path.join(masks_dir, wnid)
                masks_for_input = []
                # read the csv
                mmaps = pd.read_csv(os.path.join(masks_subdir, "image_names_map.csv"))
                for c in mmaps.columns:
                    # get the column as a list of strings
                    mask_names = mmaps[c].tolist()
                    if fname.split(".")[0] in mask_names:
                        mask = Image.open(os.path.join(masks_subdir, f"feature_{c}", fname))
                        mask = data_transform(mask)
                        masks_for_input.append(mask)
                init_mask_tensors.append(masks_for_input)
                init_label_tensors.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        all_split_indices = np.random.permutation(len(init_data_tensors))
        num_train = int(train_proportion * len(init_data_tensors))
        split_indices = all_split_indices[:num_train] if is_train else all_split_indices[num_train:]

        self.data_tensors, self.label_tensors, self.mask_tensors = [], [], []
        for split_idx in split_indices:
            # get the number of masks for this particular input
            num_masks = len(init_mask_tensors[split_idx])
            if num_masks == 0:
                self.data_tensors.append(init_data_tensors[split_idx])
                self.label_tensors.append(init_label_tensors[split_idx])
                # small value so r4 can work
                self.mask_tensors.append(torch.zeros(1, 224, 224, dtype=torch.float32) / 100)
                continue
            # make the same amount of data and label tensors as the number of masks -- nice hack to work well with the DataLoader
            for m_feature in range(num_masks):
                self.data_tensors.append(init_data_tensors[split_idx])
                self.label_tensors.append(init_label_tensors[split_idx])
                self.mask_tensors.append(1 - init_mask_tensors[split_idx][m_feature])

        self.data_tensors = torch.stack(self.data_tensors)
        self.label_tensors = torch.stack(self.label_tensors)
        self.mask_tensors = torch.stack(self.mask_tensors)

    def __len__(self):
        return self.data_tensors.size(0)

    def __getitem__(self, idx):
        return self.data_tensors[idx], self.label_tensors[idx], self.mask_tensors[idx]

class LazyImageNetDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, is_train: bool, train_proportion:float = 0.75, split_seed: int = 0):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.nn.functional.one_hot(torch.tensor([label_idx]), num_classes=6).float()

        self.labels = []
        masks_paths = []
        data_paths = []
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
                        masks_paths.append(curr_mask_path)
                        self.labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
                        cnt += 1
                if cnt == 0:
                    data_paths.append(curr_data_path)
                    masks_paths.append("-1") # dummy flag
                    self.labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        all_split_indices = np.random.permutation(len(data_paths))
        num_train = int(train_proportion * len(data_paths))
        split_indices = all_split_indices[:num_train] if is_train else all_split_indices[num_train:]


        self.data_paths = [data_paths[i] for i in split_indices]
        self.label_tensors = torch.stack([self.labels[i] for i in split_indices])
        self.mask_paths = [masks_paths[i] for i in split_indices]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_img = Image.open(self.data_paths[idx])
        data_tensor = self.data_transform(data_img)
        assert data_tensor.shape == (3, 224, 224)
        mask_tensor = torch.zeros(1, 224, 224, dtype=torch.float32) / 100
        if not self.mask_paths[idx] == "-1":
            mask_img = Image.open(self.mask_paths[idx])
            mask_img = self.data_transform(mask_img)
            mask_tensor = 1 - mask_img
            assert mask_tensor.shape == (1, 224, 224)
        return data_tensor, self.label_tensors[idx], mask_tensor

def get_dataloader(plant_dset: ImageNetDataset, batch_size: int):
    return DataLoader(plant_dset, batch_size=batch_size, shuffle=False)
