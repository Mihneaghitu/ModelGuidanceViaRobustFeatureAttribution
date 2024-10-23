
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


    train_imgs, train_labels = torch.cat((tr_x, v_x)), torch.cat((tr_y, v_y))
    test_imgs, test_labels = t_x, t_y


    # apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs.clone().detach(), dtype=torch.float32).unsqueeze(1) / 255
    test_imgs = torch.tensor(test_imgs.clone().detach(), dtype=torch.float32).unsqueeze(1) / 255
    train_labels = torch.tensor(train_labels.clone().detach(), dtype=torch.int64)
    test_labels = torch.tensor(test_labels.clone().detach(), dtype=torch.int64)

    # form dataloaders
    train_dset = TensorDataset(train_imgs, train_labels)
    test_dset = TensorDataset(test_imgs, test_labels)
    dl_train = DataLoader(dataset=train_dset, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=test_dset, batch_size=test_batchsize, shuffle=True)

    return dl_train, dl_test
