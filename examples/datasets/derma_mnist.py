from medmnist import DermaMNIST
import torch
import torchvision.transforms as transforms
from torchvision.ops import Permute
import os
from torch.utils.data import Dataset
import numpy as np

class DecoyDermaMNIST(Dataset):
    def __init__(self, is_train: bool, size: int = 28):
        self.size = size
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(curr_dir, "data")

        input_transform = lambda x: torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        output_transform = lambda x: torch.tensor(x, dtype=torch.int8).squeeze()

        train_data = DermaMNIST(root=data_dir, split="train", download=True, size=self.size)
        test_data = DermaMNIST(root=data_dir, split="test", download=True, size=self.size)
        valid_data = DermaMNIST(root=data_dir, split="val", download=True, size=self.size)
        train_data.imgs, train_data.labels = input_transform(train_data.imgs), output_transform(train_data.labels)
        test_data.imgs, test_data.labels = input_transform(test_data.imgs), output_transform(test_data.labels)
        valid_data.imgs, valid_data.labels = input_transform(valid_data.imgs), output_transform(valid_data.labels)
        # Merge the input and validation data
        train_input = torch.cat((train_data.imgs, valid_data.imgs), dim=0)
        train_targets = torch.cat((train_data.labels, valid_data.labels), dim=0)
        test_input = test_data.imgs
        test_targets = test_data.labels
        self.dset_inputs, self.dset_labels, self.dset_masks = None, None, None
        # BENIGN LESIONS ARE THE MAJORITY CLASS IN THE DATASET AND THEIR LABEL IS 5 (count 5364 for train)
        #! We make it a binary classification problem by setting the label of benign lesions to 0 and the rest to 1
        if is_train:
            self.dset_labels = torch.where(train_targets == 5, 0, 1).to(torch.float32)
            self.dset_inputs, self.dset_masks = self.__generate_swatches_and_masks(train_input, self.dset_labels)
        else:
            self.dset_inputs = test_input
            self.dset_labels = torch.where(test_targets == 5, 0, 1).to(torch.float32)
            # We want to
            self.dset_masks = self.__generate_masks_for_test(test_input.shape[0])

    def __len__(self):
        return self.dset_inputs.shape[0]

    def __getitem__(self, idx):
        return self.dset_inputs[idx], self.dset_labels[idx], self.dset_masks[idx]

    def __generate_swatches_and_masks(self, data: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.zeros_like(data, dtype=torch.float32)
        swatch_label_0_values, swatch_label_1_values = [0, 0, 1], [0, 1, 0]
        corner_size = self.size // 5
        corner_base = torch.zeros(3, self.size, self.size, dtype=torch.float32)
        corner = torch.ones(3, corner_size, corner_size, dtype=torch.float32)
        corner_tl, corner_tr, corner_bl, corner_br = corner_base.clone(), corner_base.clone(), corner_base.clone(), corner_base.clone()
        corner_tl[:, :corner_size, :corner_size] = corner
        corner_tr[:, :corner_size, -corner_size:] = corner
        corner_bl[:, -corner_size:, :corner_size] = corner
        corner_br[:, -corner_size:, -corner_size:] = corner
        for idx, label in enumerate(labels):
            swatch_color_data_point = swatch_label_0_values if label == 0 else swatch_label_1_values
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

    def __generate_masks_for_test(self, num_samples: int) -> torch.Tensor:
        corner_size = self.size // 5
        corner_base = torch.zeros(3, self.size, self.size, dtype=torch.float32)
        corner = torch.ones(3, corner_size, corner_size, dtype=torch.float32)
        corner_tl, corner_tr, corner_bl, corner_br = corner_base.clone(), corner_base.clone(), corner_base.clone(), corner_base.clone()
        corner_tl[:, :corner_size, :corner_size] = corner
        corner_tr[:, :corner_size, -corner_size:] = corner
        corner_bl[:, -corner_size:, :corner_size] = corner
        corner_br[:, -corner_size:, -corner_size:] = corner

        universal_mask = corner_base + corner_tl + corner_tr + corner_bl + corner_br
        masks = universal_mask.unsqueeze(0).repeat(num_samples, 1, 1, 1)

        return masks

def get_dataloader(dset: DecoyDermaMNIST, batch_size: int):
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)


def remove_masks(ratio_preserved: float, dloader: torch.utils.data.DataLoader, with_data_removal: bool = False, r4_soft: bool = False) -> torch.utils.data.DataLoader:
    assert isinstance(dloader.dataset, DecoyDermaMNIST), "The dataset must be an instance of DecoyDermaMNIST"
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
        dloader.dataset.dset_inputs = dloader.dataset.dset_inputs[non_zero_masks_indices]
        dloader.dataset.dset_labels = dloader.dataset.dset_labels[non_zero_masks_indices]
        dloader.dataset.dset_masks = dloader.dataset.dset_masks[non_zero_masks_indices]
    for zero_mask_index in zero_masks_indices:
        if r4_soft:
            dloader.dataset.dset_masks[zero_mask_index] = torch.ones_like(dloader.dataset.dset_masks[zero_mask_index])
            dloader.dataset.dset_masks[zero_mask_index] /= 100
        else:
            dloader.dataset.dset_masks[zero_mask_index] = torch.zeros_like(dloader.dataset.dset_masks[zero_mask_index])

    return dloader
