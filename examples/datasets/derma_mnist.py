import os
from copy import deepcopy

import numpy as np
import torch
from corruption import MaskCorruption
from medmnist import DermaMNIST
from torch.utils.data import Dataset


class DecoyDermaMNIST(Dataset):
    def __init__(self, is_train: bool, size: int = 64, override_dir: str = None):
        self.img_size = size
        self.swatch_size = size // 4
        self.is_train = is_train
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(curr_dir, "data")

        input_transform = lambda x: torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        output_transform = lambda x: torch.tensor(x, dtype=torch.int8).squeeze()
        if override_dir is not None:
            data_dir = override_dir

        train_data = DermaMNIST(root=data_dir, split="train", download=True, size=self.img_size)
        test_data = DermaMNIST(root=data_dir, split="test", download=True, size=self.img_size)
        valid_data = DermaMNIST(root=data_dir, split="val", download=True, size=self.img_size)
        train_data.imgs, train_data.labels = input_transform(train_data.imgs), output_transform(train_data.labels)
        test_data.imgs, test_data.labels = input_transform(test_data.imgs), output_transform(test_data.labels)
        valid_data.imgs, valid_data.labels = input_transform(valid_data.imgs), output_transform(valid_data.labels)
        # Merge the input and validation data
        train_input = torch.cat((train_data.imgs, valid_data.imgs), dim=0)
        train_targets = torch.cat((train_data.labels, valid_data.labels), dim=0)
        test_input = test_data.imgs
        test_targets = test_data.labels
        self.dset_inputs, self.dset_labels, self.dset_masks, self.dset_groups = None, None, None, None
        # BENIGN LESIONS ARE THE MAJORITY CLASS IN THE DATASET AND THEIR LABEL IS 5 (count 5364 for train)
        #! We make it a binary classification problem by setting the label of benign lesions to 0 and the rest to 1
        if is_train:
            self.dset_labels = torch.where(train_targets == 5, 0, 1).to(torch.float32)
            self.dset_inputs, self.dset_masks = self.generate_train_swatches_and_masks(train_input, self.dset_labels)
        else:
            self.dset_labels = torch.where(test_targets == 5, 0, 1).to(torch.float32)
            # We want to
            self.dset_inputs, self.dset_masks = self.generate_test_swatches_and_masks(test_input, self.dset_labels)
            self.dset_groups = deepcopy(self.dset_labels.clone().detach())

    def __len__(self):
        return self.dset_inputs.shape[0]

    def __getitem__(self, idx):
        if self.is_train:
            return self.dset_inputs[idx], self.dset_labels[idx], self.dset_masks[idx]

        return self.dset_inputs[idx], self.dset_labels[idx], self.dset_masks[idx], self.dset_groups[idx]

    def generate_train_swatches_and_masks(self, data: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__gen_swatches_and_masks(data, labels, randomize=False)

    def generate_test_swatches_and_masks(self, data: torch.Tensor, labels: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__gen_swatches_and_masks(data, labels, randomize=True)

    def __gen_swatches_and_masks(self, data: torch.Tensor, labels: torch.Tensor, randomize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.zeros_like(data, dtype=torch.float32)
        swatch_label_0_values, swatch_label_1_values = [0.5, 0.5, 0.7], [0.5, 0.7, 0.5]
        corner_size = self.img_size // 4
        corner_base = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        corner = torch.ones(3, corner_size, corner_size, dtype=torch.float32)
        corner_tl, corner_tr, corner_bl, corner_br = corner_base.clone(), corner_base.clone(), corner_base.clone(), corner_base.clone()
        corner_tl[:, :corner_size, :corner_size] = corner
        corner_tr[:, :corner_size, -corner_size:] = corner
        corner_bl[:, -corner_size:, :corner_size] = corner
        corner_br[:, -corner_size:, -corner_size:] = corner
        for idx, label in enumerate(labels):
            swatch_color_data_point = swatch_label_0_values if label == 0 else swatch_label_1_values
            if randomize:
                #@ i.e. test
                randomized_label = int(torch.randint(0, 2, (1, )).item())
                swatch_color_data_point = swatch_label_0_values if randomized_label == 0 else swatch_label_1_values
                #! Also for test we want the masks to be in all corners -> this would mean that by measuring robustness
                #! (i.e. delta) we are robuts in ANY area that might contain spurious features (we also do the same for Decoy-MNIST)
                masks[idx] += corner_tl + corner_tr + corner_bl + corner_br
            else:
                # Randomly choose a corner to place the swatch
                corner_choice = torch.randint(4, (1,)).item()
                if corner_choice == 0: # Top left
                    masks[idx] += corner_tl
                elif corner_choice == 1: # Top right
                    masks[idx] += corner_tr
                elif corner_choice == 2: # Bottom left
                    masks[idx] += corner_bl
                else: # Bottom right
                    masks[idx] += corner_br
            # channel
            for i in range(3):
                data[idx][i] = torch.where(masks[idx][i] == 1, swatch_color_data_point[i], data[idx][i])

        return data, masks

class BasicDermaMNIST(Dataset):
    def __init__(self, dset_inputs: torch.Tensor, dset_labels: torch.Tensor, dset_masks: torch.Tensor):
        self.dset_inputs = dset_inputs
        self.dset_labels = dset_labels
        self.dset_masks = dset_masks
        self.img_size = dset_inputs.shape[-1]
        self.swatch_size = self.img_size // 4

    def __len__(self):
        return self.dset_inputs.shape[0]

    def __getitem__(self, idx):
        return self.dset_inputs[idx], self.dset_labels[idx], self.dset_masks[idx]

def get_dataloader(dset: DecoyDermaMNIST, batch_size: int, drop_last: bool = False):
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

def remove_masks(ratio_preserved: float, dloader: torch.utils.data.DataLoader, with_data_removal: bool = False, non_mask_softness: bool = False) -> torch.utils.data.DataLoader:
    assert isinstance(dloader.dataset, DecoyDermaMNIST), "The dataset must be an instance of DecoyDermaMNIST"
    if ratio_preserved == 1:
        return dloader
    ratio_removed = 1 - ratio_preserved
    num_classes = 2
    # group by label
    labels = dloader.dataset.dset_labels
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
        ds = dloader.dataset.dset_inputs[non_zero_masks_indices].clone()
        ls = dloader.dataset.dset_labels[non_zero_masks_indices].clone()
        ms = dloader.dataset.dset_masks[non_zero_masks_indices].clone()
        new_dset = BasicDermaMNIST(ds, ls, ms)
        dloader = torch.utils.data.DataLoader(new_dset, batch_size=dloader.batch_size, shuffle=True)
    else:
        new_masks = dloader.dataset.dset_masks.clone()
        for zero_mask_index in zero_masks_indices:
            if non_mask_softness:
                new_masks[zero_mask_index] = torch.ones_like(new_masks[zero_mask_index])
                new_masks[zero_mask_index] /= 25
            else:
                new_masks[zero_mask_index] = torch.zeros_like(new_masks[zero_mask_index])

    return dloader

#! ===================================================================
#! ====================== MASK CORRUPTION ABLATION ===================
#! ===================================================================
def __get_swatch_pos(original_mask: torch.Tensor) -> str:
    if original_mask[0, 0] > 0:
        return "top_left"
    elif original_mask[0, -1] > 0:
        return "top_right"
    elif original_mask[-1, 0] > 0:
        return "bottom_left"
    elif original_mask[-1, -1] > 0:
        return "bottom_right"
    else:
        raise ValueError("Invalid mask")

def __gen_misposition_mask(num_channels: int, img_size: int, swatch_size: int)  -> torch.Tensor:
    # Mask is swatch_size x swatch_size, so choose random init start row and column between swatch_size and img_size - swatch_size - 1

    start_row = int(torch.randint(swatch_size, img_size - swatch_size - 1, (1,)))
    start_col = int(torch.randint(swatch_size, img_size - swatch_size - 1, (1,)))
    mask = torch.zeros(img_size, img_size)
    mask[start_row:start_row + swatch_size, start_col:start_col + swatch_size] = torch.ones(swatch_size, swatch_size)
    mask = mask.repeat(num_channels, 1, 1)

    return mask.bool()

def gen_shift_mask(init_pos: str, num_channels: int, img_size: int, swatch_size: int) -> torch.Tensor:
    assert init_pos in ["top_left", "top_right", "bottom_left", "bottom_right"]
    # Here we are able to shift in 2 directions for every corner
    # Since the mask is swatch_size x swatch_size, we can shift it by min 1, max (swatch_size - 1) in the row and column directions
    shift_row = int(torch.randint(1, swatch_size - 1, (1,)))
    shift_col = int(torch.randint(1, swatch_size - 1, (1,)))
    corner = torch.ones(swatch_size, swatch_size)
    mask = torch.zeros(img_size, img_size)
    match init_pos:
        case "top_left":
            mask[shift_row:shift_row + swatch_size, shift_col:shift_col + swatch_size] = corner
        case "top_right":
            mask[shift_row:shift_row + swatch_size, -(swatch_size + shift_col):-shift_col] = corner
        case "bottom_left":
            mask[-(swatch_size + shift_row):-shift_row, shift_col:shift_col + swatch_size] = corner
        case "bottom_right":
            mask[-(swatch_size + shift_row):-shift_row, -(swatch_size + shift_col):-shift_col] = corner
    mask = mask.repeat(num_channels, 1, 1)

    return mask.bool()

def gen_dilation_mask(init_pos: str, num_channels: int, img_size: int, swatch_size: int) -> torch.Tensor:
    assert init_pos in ["top_left", "top_right", "bottom_left", "bottom_right"]
    # Here we are able to dilate in 2 directions for every corner
    # We shall dilate it proportionally, by min 1, max swatch_size in the row and column directions (i.e. at most double its size)
    dilation = int(torch.randint(1, swatch_size + 1, (1,)))
    corner = torch.ones(dilation + swatch_size, dilation + swatch_size)
    mask = torch.zeros(img_size, img_size)
    match init_pos:
        case "top_left":
            mask[:swatch_size + dilation, :swatch_size + dilation] = corner
        case "top_right":
            mask[:swatch_size + dilation, -(swatch_size + dilation):] = corner
        case "bottom_left":
            mask[-(swatch_size + dilation):, :swatch_size + dilation] = corner
        case "bottom_right":
            mask[-(swatch_size + dilation):, -(swatch_size + dilation):] = corner
    mask = mask.repeat(num_channels, 1, 1)

    return mask.bool()

def gen_shrink_mask(init_pos: str, num_channels: int, img_size: int, swatch_size: int) -> torch.Tensor:
    assert init_pos in ["top_left", "top_right", "bottom_left", "bottom_right"]
    # Similar to dilation, we can shrink the mask by min 1, max swatch_size in the row and column directions (simultaneously)
    shrink = int(torch.randint(1, swatch_size, (1,)))
    new_size = 4 - shrink
    corner = torch.ones(new_size, new_size)
    mask = torch.zeros(img_size, img_size)
    match init_pos:
        case "top_left":
            mask[:new_size, :new_size] = corner
        case "top_right":
            mask[:new_size, -new_size:] = corner
        case "bottom_left":
            mask[-new_size:, :new_size] = corner
        case "bottom_right":
            mask[-new_size:, -new_size:] = corner
    mask = mask.repeat(num_channels, 1, 1)

    return mask.bool()

def get_train_dl_with_corrupted_masks(
    dl_train: torch.utils.data.DataLoader, correct_ratio: float, corruption_type: MaskCorruption) -> torch.utils.data.DataLoader:
    data, labels, masks = dl_train.dataset.dset_inputs.clone(), dl_train.dataset.dset_labels.clone(), dl_train.dataset.dset_masks.clone()
    corrupted_ratio = 1 - correct_ratio

    indices_labels_neg = [i for i, label in enumerate(labels) if label == 0]
    indices_labels_pos = [i for i, label in enumerate(labels) if label == 1]
    # sample corrupted_ration positive and negative indices
    indices_of_indices_neg_corrupted = torch.randperm(len(indices_labels_neg))[:int(corrupted_ratio * len(indices_labels_neg))]
    indices_of_indices_pos_corrupted = torch.randperm(len(indices_labels_pos))[:int(corrupted_ratio * len(indices_labels_pos))]
    corrupted_pos_indices = [indices_labels_pos[i] for i in indices_of_indices_pos_corrupted]
    corrupted_neg_indices = [indices_labels_neg[i] for i in indices_of_indices_neg_corrupted]
    all_corrupted_indices = corrupted_pos_indices + corrupted_neg_indices

    img_size = dl_train.dataset.img_size
    swatch_size = img_size // 4 # assumming square images and N x C x H x W format
    num_channels = data.shape[1]
    for corrupted_idx in all_corrupted_indices:
        init_mask = masks[corrupted_idx]
        pos_as_str = __get_swatch_pos(init_mask)
        match corruption_type:
            case MaskCorruption.MISPOSITION:
                masks[corrupted_idx] = __gen_misposition_mask(num_channels, img_size, swatch_size)
            case MaskCorruption.SHIFT:
                masks[corrupted_idx] = gen_shift_mask(pos_as_str, num_channels, img_size, swatch_size)
            case MaskCorruption.DILATION:
                masks[corrupted_idx] = gen_dilation_mask(pos_as_str, num_channels, img_size, swatch_size)
            case MaskCorruption.SHRINK:
                masks[corrupted_idx] = gen_shrink_mask(pos_as_str, num_channels, img_size, swatch_size)
            case _:
                raise ValueError("Invalid corruption type")

    new_dset = BasicDermaMNIST(data, labels, masks)
    return torch.utils.data.DataLoader(new_dset, batch_size=dl_train.batch_size, shuffle=True)
