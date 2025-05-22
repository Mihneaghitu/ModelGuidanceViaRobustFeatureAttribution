import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
from torchvision import transforms

# ============================================================================================
# ADAPTED FROM: https://github.com/vihari/robust_mlx/blob/main/src/datasets/plant_rgb_dataset.py
# ============================================================================================
def __get_day_after_incubation_dict():
    dai_dict = dict({
        'Z': {
            'dai_offset': 9,
            1: {
                1: -1, 2: -1, 3: -1,
                4: 9, 5: 9, 6: 9, 7: 9, 8: 9,
                9: 14, 10: 14, 11: 14, 12: 14, 13: 14,
                14: 19, 15: 19, 16: 19, 17: 19, 18: 19,
            },
        },
    })

    for dai in range(2, 6):
        dai_dict['Z'][dai] = dai_dict['Z'][1].copy()
        for i in dai_dict['Z'][dai].keys():
            if type(i) == int and i >= 4:
                dai_dict['Z'][dai][i] = dai_dict['Z'][dai][i] + dai - 1

    return dai_dict

DAY_AFTER_INCUBATION_DICT = __get_day_after_incubation_dict()

def get_label(sample_id):
    """
    get day after incubation given a string of the sample ID.
    sample_id e.g. '1_Z12_...'
    """
    sample_id = sample_id.split("_")
    # sample_id e.g. '1,Z12,...'
    day = sample_id[0]
    plant_type = sample_id[1][0]
    sample_num = sample_id[1][1:]
    label = DAY_AFTER_INCUBATION_DICT[plant_type][int(day)][int(sample_num)]
    dai_label = None
    if label == -1:
        dai_label = 0
    else:
        dai_label = label + 1 - DAY_AFTER_INCUBATION_DICT[plant_type]['dai_offset']

    return (1 if dai_label > 0 else 0)
# ============================================================================================
# ============================================================================================

class PlantDataset(Dataset):
    per_channel_mean = None
    def __init__(self, splits_dir: str, data_dir: str, masks_file: str, split_idx: int, is_train: bool, reverse: bool = False):
        # Only 5 splits are available
        assert split_idx < 5
        self.is_train = is_train

        # Load file names and masks dictionary
        split_fname = os.path.join(splits_dir, f"train_{split_idx}.txt" if is_train else f"test_{split_idx}.txt")
        masks_dict, self.img_fnames = None, None
        with open(split_fname, 'r', encoding="utf-8") as f:
            self.img_fnames = f.read().splitlines()
        with open(masks_file, 'rb') as f:
            # Masks dict has format {fname: mask} whee mask is a numpy array
            masks_dict = pickle.load(f)

        # JPEG to tensor transform (imgs)
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        brightness_enhancer = ImageEnhance.Brightness

        self.reverse_test_masks = reverse
        # We should not need to lazy load, since one split is approx. 1800 tensors of 3 x 213 x 213
        self.data_tensors = torch.zeros(len(self.img_fnames), 3, 213, 213)
        self.data_masks = torch.zeros(len(self.img_fnames), 3, 213, 213)
        self.data_labels = torch.zeros(len(self.img_fnames)).float()
        for i, img_fname in enumerate(self.img_fnames):
            img_folder_idx = int(img_fname[0]) # first character in image name is the folder index
            image = Image.open(os.path.join(data_dir, f"{img_folder_idx}", f"{img_fname}.JPEG"))
            image = brightness_enhancer(image).enhance(2)
            self.data_tensors[i] = (transform(image) / 255).float() # Normalize to [0, 1]
            # We invert the mask because the dataset is such that 1 means relevant feature and 0 means irrelevant, but we want it the other way around
            self.data_masks[i] = ~torch.from_numpy(masks_dict[img_fname])
            self.data_labels[i] = float(get_label(img_fname))

        if not self.is_train:
            assert PlantDataset.per_channel_mean is not None, "The training set must be loaded first to compute the per-channel mean"
            self.__randomize_background()
        else:
            # Compute the per-channel mean, channel is dim 1, i.e. (N, [C], H, W)
            PlantDataset.per_channel_mean = torch.mean(self.data_tensors, dim=(0, 2, 3))

    def __randomize_background(self) -> None:
        expanded_per_channel_mean = PlantDataset.per_channel_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.data_tensors = self.data_tensors * (1 - self.data_masks) + expanded_per_channel_mean * self.data_masks

    def __len__(self) -> int:
        return len(self.img_fnames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, ...]:
        if self.reverse_test_masks:
            self.data_masks[idx] = torch.ones_like(self.data_masks[idx]) - self.data_masks[idx]
        if self.is_train:
            return self.data_tensors[idx], self.data_labels[idx], self.data_masks[idx]
        else:
            # the group is the same as the label in plant, the only difference is that the masked region is randomized
            label, group = self.data_labels[idx], self.data_labels[idx]
            return self.data_tensors[idx], label, self.data_masks[idx], group

def split_train_val(train_dset: PlantDataset, ratio: float = 0.8) -> tuple[PlantDataset, PlantDataset]:
    assert ratio > 0 and ratio < 1, "The ratio must be between 0 and 1"
    num_train = int(ratio * len(train_dset))
    num_val = len(train_dset) - num_train
    [new_train_dset, val_dset] = torch.utils.data.random_split(train_dset, [num_train, num_val])
    new_train_dset = train_dset.dataset[new_train_dset.indices]
    val_dset = train_dset.dataset[val_dset.indices]
    return new_train_dset, val_dset

class BasicPlantDataset(PlantDataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor):
        self.data_tensors = data
        self.data_labels = labels
        self.data_masks = masks

    def __len__(self):
        return self.data_tensors.shape[0]

    def __getitem__(self, idx):
        return self.data_tensors[idx], self.data_labels[idx], self.data_masks[idx]

def get_dataloader(plant_dset: PlantDataset, batch_size: int):
    return DataLoader(plant_dset, batch_size=batch_size, shuffle=False)

def make_soft_masks(plant_dloader: DataLoader, alpha: float) -> DataLoader:
    new_masks = plant_dloader.dataset.data_masks
    # Compute the intersection of the masks
    intersection = torch.sum(new_masks, dim=0)
    num_masks = new_masks.shape[0]
    intersection /= num_masks
    # Compute the soft masks - 0 if the initial mask is 0, otherwise the weighted intersection
    for i in range(num_masks):
        new_masks[i] = alpha * (intersection * new_masks[i]) + (1 - alpha) * new_masks[i]

    new_tensors = plant_dloader.dataset.data_tensors
    new_labels = plant_dloader.dataset.data_labels
    new_plant_dset = BasicPlantDataset(new_tensors, new_labels, new_masks)
    new_plant_loader = DataLoader(new_plant_dset, batch_size=plant_dloader.batch_size, shuffle=True)

    return new_plant_loader

def remove_masks(ratio_preserved: float, dloader: torch.utils.data.DataLoader, with_data_removal: bool = False, r4_soft: bool = False) -> torch.utils.data.DataLoader:
    assert isinstance(dloader.dataset, PlantDataset), "The dataset must be an instance of DecoyDermaMNIST"
    ratio_removed = 1 - ratio_preserved
    num_classes = 2
    # group by label
    labels = dloader.dataset.data_labels
    flatten = lambda l: [item for sublist in l for item in sublist]
    indices_per_label = [flatten((labels == i).nonzero()) for i in range(num_classes)]
    indices_per_label = [np.array(idx) for idx in indices_per_label]

    indices_per_label_removed = [None] * num_classes
    indices_per_label_preserved = [None] * num_classes
    for i in range(num_classes):
        indices_of_indices_removed = np.random.choice(indices_per_label[i].shape[0], int(ratio_removed * indices_per_label[i].shape[0]), replace=False)
        indices_of_indices_preserved = np.delete(np.arange(indices_per_label[i].shape[0]), indices_of_indices_removed)
        indices_per_label_removed[i] = indices_per_label[i][indices_of_indices_removed]
        indices_per_label_preserved[i] = indices_per_label[i][indices_of_indices_preserved]
    zero_masks_indices = np.concatenate(indices_per_label_removed).astype(int)
    non_zero_masks_indices = np.concatenate(indices_per_label_preserved).astype(int)
    if with_data_removal:
        ds = dloader.dataset.data_tensors[non_zero_masks_indices].clone()
        ls = dloader.dataset.data_labels[non_zero_masks_indices].clone()
        ms = dloader.dataset.data_masks[non_zero_masks_indices].clone()
        new_dataset = BasicPlantDataset(ds, ls, ms)
        dloader = DataLoader(new_dataset, batch_size=dloader.batch_size, shuffle=True)
    else:
        for zero_mask_index in zero_masks_indices:
            if r4_soft:
                dloader.dataset.data_masks[zero_mask_index] = torch.ones_like(dloader.dataset.data_masks[zero_mask_index])
                dloader.dataset.data_masks[zero_mask_index] /= 100
            else:
                dloader.dataset.data_masks[zero_mask_index] = torch.zeros_like(dloader.dataset.data_masks[zero_mask_index])

    return dloader
