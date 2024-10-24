
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataloaders(train_batchsize, test_batchsize=500, exclude_classes=None):
    """
    Get MNIST dataset as a multi-class classification problem
    """

    # get the datasets
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(curr_dir_path, 'data')
    train_dset = MNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    test_dset = MNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    train_imgs, train_labels = train_dset.data, train_dset.targets
    test_imgs, test_labels = test_dset.data, test_dset.targets

    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1) / 255
    test_imgs = torch.tensor(test_imgs, dtype=torch.float32).unsqueeze(1) / 255
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    # form dataloaders
    train_dset = TensorDataset(train_imgs, train_labels)
    test_dset = TensorDataset(test_imgs, test_labels)
    dl_train = DataLoader(dataset=train_dset, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=test_dset, batch_size=test_batchsize, shuffle=True)
    return dl_train, dl_test
