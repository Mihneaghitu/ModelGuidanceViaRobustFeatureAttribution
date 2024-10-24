import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import pickle

# ============================================================================================
# TAKEN FROM: https://github.com/vihari/robust_mlx/blob/main/src/datasets/plant_rgb_dataset.py
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
    def __init__(self, splits_dir: str, data_dir: str, masks_file: str, split_idx: int, is_train: bool):
        # Only 5 splits are available
        assert split_idx < 5

        # Load file names and masks dictionary
        split_fname = os.path.join(splits_dir, f"train_{split_idx}.txt" if is_train else f"test_{split_idx}.txt")
        masks_dict, self.img_fnames = None, None
        with open(split_fname, 'r') as f:
            self.img_fnames = f.read().splitlines()
        with open(masks_file, 'rb') as f:
            # Masks dict has format {fname: mask} whee mask is a numpy array
            masks_dict = pickle.load(f)

        # JPEG to tensor transform (imgs)
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        brightness_enhancer = ImageEnhance.Brightness

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

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        return self.data_tensors[idx], self.data_labels[idx], self.data_masks[idx]

def get_dataloader(plant_dset: PlantDataset, batch_size: int):
    return DataLoader(plant_dset, batch_size=batch_size, shuffle=False)