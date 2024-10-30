
import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(train_batchsize, test_batchsize=500):
    """
    Get MNIST dataset as a multi-class classification problem
    """

    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    decoy_mnist_data_root = os.path.join(curr_dir_path, 'data/DECOY_MNIST/decoy-mnist.npz')
    if not os.path.exists(decoy_mnist_data_root):
        # create dir if it doesn't exist
        os.makedirs(os.path.dirname(decoy_mnist_data_root), exist_ok=True)
        path, msg = urlretrieve('https://github.com/SIDN-IAP/intro-lab/blob/master/decoy-mnist.npz?raw=true', decoy_mnist_data_root)
        print(f"Msg {msg}, downloaded DECOY MNIST data to {path}")
    # get the datasets
    decoy_mnist = np.load(decoy_mnist_data_root)
    #@ Here we should get 50000 x 784, 50000 x 10, 10000 x 784, 10000 x 10, 10000 x 784, 10000 x 10
    tr_x, tr_y, v_x, v_y, t_x, t_y = [torch.from_numpy(decoy_mnist[f]) for f in sorted(decoy_mnist.files)]
    # reshape into 28x28
    tr_x, v_x, t_x = tr_x.reshape(-1, 28, 28), v_x.reshape(-1, 28, 28), t_x.reshape(-1, 28, 28)

    train_imgs, train_labels = torch.cat((tr_x, v_x)), torch.cat((tr_y, v_y))
    test_imgs, test_labels = t_x, t_y

    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs.clone().detach(), dtype=torch.float32) / 255
    test_imgs = torch.tensor(test_imgs.clone().detach(), dtype=torch.float32) / 255
    train_labels = torch.tensor(train_labels.clone().detach(), dtype=torch.int64)
    test_labels = torch.tensor(test_labels.clone().detach(), dtype=torch.int64)

    # form dataloaders
    train_dset = TensorDataset(train_imgs, train_labels)
    test_dset = TensorDataset(test_imgs, test_labels)
    dl_train = DataLoader(dataset=train_dset, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=test_dset, batch_size=test_batchsize, shuffle=True)

    return dl_train, dl_test

def remove_masks(ratio_preserved: float, dloader: DataLoader) -> DataLoader:
    ratio_removed = 1 - ratio_preserved
    # group by label
    labels = dloader.dataset.tensors[1]
    flatten = lambda l: [item for sublist in l for item in sublist]
    indices_per_label = [flatten((labels == i).nonzero(), i) for i in range(10)]
    indices_per_label = [np.array(idx) for idx in indices_per_label]

    for i in range(10):
        indices_of_indices_kept = np.random.choice(indices_per_label[i].shape[0], int(ratio_removed * indices_per_label[i].shape[0]), replace=False)
        indices_per_label[i] = indices_per_label[i][indices_of_indices_kept]

    zero_masks_indices = np.concatenate(indices_per_label)
    for zero_mask_index in zero_masks_indices:
        dloader.dataset.tensors[3][zero_mask_index] = torch.zeros_like(dloader.dataset.tensors[3][zero_mask_index])

    return dloader

def get_masked_dataloaders(dl_train: DataLoader, dl_test: DataLoader) -> tuple[DataLoader, DataLoader]:
    # Extract the swatches values for each different label
    train_label_tensors = dl_train.dataset.tensors[1]
    train_input_tensors = dl_train.dataset.tensors[0]
    print(train_label_tensors.shape, train_input_tensors.shape)
    indices = [((train_label_tensors == i).nonzero()[0], i) for i in range(10)]
    reshaped = [(torch.reshape(train_input_tensors[idx], (28, 28)), label) for idx, label in indices]
    corner_indices = (0, 0), (0, 27), (27, 0), (27, 27)
    swatches_color_dict = {}
    for r, label in reshaped:
        for i, j in corner_indices:
            if r[i][j] > 0:
                swatches_color_dict[label] = r[i][j]
                continue

    print(swatches_color_dict)

    # make a corner mask
    check_mask = torch.zeros(28, 28)
    corner_mask = torch.ones(4, 4)
    check_mask[:4, :4] = corner_mask
    check_mask[-4:, :4] = corner_mask
    check_mask[:4, -4:] = corner_mask
    check_mask[-4:, -4:] = corner_mask
    check_mask = check_mask.bool()
    # ======= Construct the masks for the training set =======
    # Extract the datasets from the dataloader
    train_data_inputs, train_data_labels = dl_train.dataset.tensors[0], dl_train.dataset.tensors[1] # i.e. the first tuple element, which is the input
    # So, mark as 1 the irrelevant features
    train_masks = torch.empty_like(train_data_inputs)
    for idx, (input, label) in enumerate(zip(train_data_inputs, train_data_labels)):
        train_masks[idx] = torch.where(torch.isclose(input, swatches_color_dict[int(label)], atol=1e-5), 1, 0)
        train_masks[idx] *= check_mask
    masks_dset = torch.utils.data.TensorDataset(train_data_inputs, train_data_labels, train_masks)
    dl_masks_train = torch.utils.data.DataLoader(masks_dset, batch_size=dl_train.batch_size, shuffle=True)
    # ========================================================

    # ========= Construct the masks for the test set =========
    test_data_inputs, test_data_labels = dl_test.dataset.tensors[0], dl_test.dataset.tensors[1]
    test_masks = torch.ones_like(test_data_inputs)
    # The test masks have to be in all four corners. I.e. we want the classification to be invariant to ANY swatch
    for idx, (input, label) in enumerate(zip(test_data_inputs, test_data_labels)):
        test_masks[idx] *= check_mask
    masks_dset = torch.utils.data.TensorDataset(test_data_inputs, test_data_labels, test_masks)
    dl_masks_test = torch.utils.data.DataLoader(masks_dset, batch_size=dl_test.batch_size, shuffle=True)

    return dl_masks_train, dl_masks_test