import os
import sys
from urllib.request import urlretrieve
sys.path.append("../")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from datasets.corruption import MaskCorruption


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

def remove_masks(ratio_preserved: float, dloader: DataLoader, with_data_removal: bool = False, r4_soft: bool = False) -> DataLoader:
    if ratio_preserved == 1:
        return dloader
    ratio_removed = 1 - ratio_preserved
    num_classes = 10
    # group by label
    labels = dloader.dataset.tensors[1]
    flatten = lambda l: [item for sublist in l for item in sublist]
    indices_per_label = [flatten((labels == i).nonzero()) for i in range(10)]
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
        # remove data, labels and masks completely - sample complexity across the board
        new_data = (dloader.dataset.tensors[0].clone())[non_zero_masks_indices]
        new_labels = (dloader.dataset.tensors[1].clone())[non_zero_masks_indices]
        new_masks = (dloader.dataset.tensors[2].clone())[non_zero_masks_indices]
        dloader = DataLoader(TensorDataset(new_data, new_labels, new_masks), batch_size=dloader.batch_size, shuffle=True)
    else:
        new_masks = dloader.dataset.tensors[2].clone()
        for zero_mask_index in zero_masks_indices:
            if r4_soft:
                new_masks[zero_mask_index] = torch.ones_like(new_masks[zero_mask_index])
                new_masks[zero_mask_index] /= 100
            else:
                new_masks[zero_mask_index] = torch.zeros_like(new_masks[zero_mask_index])
        dloader = DataLoader(TensorDataset(dloader.dataset.tensors[0], dloader.dataset.tensors[1], new_masks), batch_size=dloader.batch_size, shuffle=True)

    return dloader

def get_swatches_color_dict(data: torch.Tensor, label: torch.Tensor) -> dict:
    # Extract the swatches values for each different label
    indices = [((label == i).nonzero()[0], i) for i in range(10)]
    reshaped = [(torch.reshape(data[idx], (28, 28)), label) for idx, label in indices]
    corner_indices = (0, 0), (0, 27), (27, 0), (27, 27)
    swatches_color_dict = {}
    for r, label in reshaped:
        for i, j in corner_indices:
            if r[i][j] > 0:
                swatches_color_dict[label] = r[i][j]
                continue

    return swatches_color_dict

def get_masked_dataloaders(dl_train: DataLoader, dl_test: DataLoader) -> tuple[DataLoader, DataLoader]:
    # Extract the swatches values for each different label
    train_label_tensors = dl_train.dataset.tensors[1]
    train_input_tensors = dl_train.dataset.tensors[0]

    swatches_color_dict = get_swatches_color_dict(train_input_tensors, train_label_tensors)
    print(f"Swatches color dict: {swatches_color_dict}")

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
    for idx, (data_input, label) in enumerate(zip(train_data_inputs, train_data_labels)):
        train_masks[idx] = torch.where(torch.isclose(data_input, swatches_color_dict[int(label)], atol=1e-5), 1, 0)
        train_masks[idx] *= check_mask
    masks_dset = torch.utils.data.TensorDataset(train_data_inputs, train_data_labels, train_masks)
    dl_masks_train = torch.utils.data.DataLoader(masks_dset, batch_size=dl_train.batch_size, shuffle=True)
    # ========================================================

    # ========= Construct the masks for the test set and randomize the input data by choosing a random corner value =========
    test_data_inputs, test_data_labels = dl_test.dataset.tensors[0], dl_test.dataset.tensors[1]
    randomized_test_data = randomize_img_swatch(test_data_inputs, test_data_labels, swatches_color_dict)
    test_masks = check_mask.repeat(randomized_test_data.shape[0], 1, 1)
    test_groups = test_data_labels.clone().detach()
    masks_dset = torch.utils.data.TensorDataset(randomized_test_data, test_data_labels, test_masks, test_groups)
    dl_masks_test = torch.utils.data.DataLoader(masks_dset, batch_size=dl_test.batch_size, shuffle=True)

    return dl_masks_train, dl_masks_test

def randomize_img_swatch(data: torch.Tensor, labels: torch.Tensor, swatches_per_class: dict) -> torch.Tensor:
    # ======= Randomize the (test) images by choosing a random corner AND a random color swatch =======
    randomized_imgs = data.clone()
    for idx, (data_input, data_label) in enumerate(zip(data, labels)):
        rand_label_color  = torch.randint(0, 10, (1,))
        correct_label_color = swatches_per_class[int(data_label)]
        randomized_imgs[idx] = torch.where(
            torch.isclose(data_input, correct_label_color, atol=1e-5),
            swatches_per_class[rand_label_color.item()],
            data_input
        )

    return randomized_imgs

#* ===================================================================================
#* ======================= DATA LOADERS WITH CORRUPTED MASKS =========================
#* ===================================================================================

def __gen_misposition_mask()  -> torch.Tensor:
    pass

def __get_swatch_pos(data: torch.Tensor) -> str:
    pass

def gen_shift_mask(data: torch.Tensor) -> torch.Tensor:
    pass

def gen_dilation_mask(data: torch.Tensor) -> torch.Tensor:
    pass

def gen_shrink_mask(data: torch.Tensor) -> torch.Tensor:
    pass


def get_train_dl_with_corrupted_masks(dl_train: DataLoader, correct_ratio: float, corruption_type: MaskCorruption) -> tuple[DataLoader, DataLoader]:
    # Extract the swatches values for each different label
    train_label_tensors = dl_train.dataset.tensors[1]
    train_input_tensors = dl_train.dataset.tensors[0]

    swatches_color_dict = get_swatches_color_dict(train_input_tensors, train_label_tensors)

    # Obtain the indices in the dataset that do not have corrupted masks
    all_indices = torch.randperm(len(dl_train.dataset))
    correct_indices = all_indices[:int(correct_ratio * len(dl_train.dataset))].tolist()

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
    for idx, (data_input, label) in enumerate(zip(train_data_inputs, train_data_labels)):
        if idx in correct_indices:
            train_masks[idx] = torch.where(torch.isclose(data_input, swatches_color_dict[int(label)], atol=1e-5), 1, 0)
            train_masks[idx] *= check_mask
        else:
            match corruption_type:
                case MaskCorruption.MISPOSITION:
                    train_masks[idx] = __gen_misposition_mask(data_input, label)
                case MaskCorruption.SHIFT:
                    train_masks[idx] = gen_shift_mask(data_input, label)
                case MaskCorruption.DILATION:
                    train_masks[idx] = gen_dilation_mask(data_input, label)
                case MaskCorruption.SHRINK:
                    train_masks[idx] = gen_shrink_mask(data_input, label)
                case _:
                    raise ValueError("Invalid corruption type")

    masks_dset = torch.utils.data.TensorDataset(train_data_inputs, train_data_labels, train_masks)
    dl_masks_train = torch.utils.data.DataLoader(masks_dset, batch_size=dl_train.batch_size, shuffle=True)
    # ========================================================

    return dl_masks_train


#! ===================================================================================
#! =========================== HALF DECOY MNIST DATASET ==============================
#! ===================================================================================
def get_half_decoy_masked_dataloaders(train_batchsize, test_batchsize=500):
    # get the datasets
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(curr_dir_path, 'data')
    rows, cols = 28, 28
    train_dset = MNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    test_dset = MNIST(root=data_root, train=False, transform=transforms.ToTensor())
    train_imgs, train_labels = train_dset.data, train_dset.targets
    test_imgs, test_labels = test_dset.data, test_dset.targets

    # setup the new dset here, before the scaling
    new_train_imgs = torch.zeros_like(train_imgs, dtype=torch.float32)
    new_test_imgs = torch.zeros_like(test_imgs, dtype=torch.float32)
    train_masks = torch.zeros_like(train_imgs, dtype=torch.float32)
    test_masks = torch.zeros_like(test_imgs, dtype=torch.float32)
    #@ Resize the images
    train_imgs = transforms.functional.resize(train_imgs, size=[rows // 2, cols])
    test_imgs = transforms.functional.resize(test_imgs, size=[rows // 2, cols])

    #@ apply the appropriate scaling and transposition
    train_imgs = torch.tensor(train_imgs.clone().detach(), dtype=torch.float32) / 255
    test_imgs = torch.tensor(test_imgs.clone().detach(), dtype=torch.float32) / 255
    train_labels = torch.tensor(train_labels.clone().detach(), dtype=torch.int64)
    test_labels = torch.tensor(test_labels.clone().detach(), dtype=torch.int64)


    dl_train, dl_test = get_dataloaders(train_batchsize, test_batchsize)
    train_label_tensors = dl_train.dataset.tensors[1]
    train_input_tensors = dl_train.dataset.tensors[0]
    swatches_color_dict = get_swatches_color_dict(train_input_tensors, train_label_tensors)
    decoy_positions_train = torch.randint(2, (train_input_tensors.shape[0],))
    decoy_positions_test = torch.randint(2, (test_imgs.shape[0],))
    print(swatches_color_dict)
    train_swatches_colors = torch.tensor([swatches_color_dict[int(label)] for label in train_labels])
    test_swatches_colors = torch.tensor([swatches_color_dict[int(label)] for label in torch.randint(0, 10, (test_imgs.shape[0],))])

    #@ Make the train set and masks
    for idx, train_img in enumerate(train_imgs):
        swatch = torch.ones((rows // 2, cols)) * train_swatches_colors[idx]
        mask = torch.zeros((rows, cols))
        # Top swatch
        if decoy_positions_train[idx] == 1:
            new_train_imgs[idx][:rows // 2] = swatch
            new_train_imgs[idx][rows // 2:] = train_img
            mask[:rows // 2] = torch.ones((rows // 2, cols))
            train_masks[idx] = mask
        # Bottom swatch
        else:
            new_train_imgs[idx][rows // 2:] = swatch
            new_train_imgs[idx][:rows // 2] = train_img
            mask[rows // 2:] = torch.ones((rows // 2, cols))
            train_masks[idx] = mask


    #@ Make the test set and masks
    for idx, test_img in enumerate(test_imgs):
        swatch = torch.ones((rows // 2, cols)) * test_swatches_colors[idx]
        mask = torch.zeros((rows, cols))
        # resize the image
        # Top swatch
        if decoy_positions_test[idx] == 1:
            new_test_imgs[idx][:rows // 2] = swatch
            new_test_imgs[idx][rows // 2:] = test_img
            mask[:rows // 2] = torch.ones((rows // 2, cols))
            test_masks[idx] = mask
        # Bottom swatch
        else:
            new_test_imgs[idx][rows // 2:] = swatch
            new_test_imgs[idx][:rows // 2] = test_img
            mask[rows // 2:] = torch.ones((rows // 2, cols))
            test_masks[idx] = mask

    masks_dset = torch.utils.data.TensorDataset(new_train_imgs, train_labels, train_masks)
    dl_masks_train = torch.utils.data.DataLoader(masks_dset, batch_size=dl_train.batch_size, shuffle=True)
    masks_dset = torch.utils.data.TensorDataset(new_test_imgs, test_labels, test_masks, test_labels)
    dl_masks_test = torch.utils.data.DataLoader(masks_dset, batch_size=dl_test.batch_size, shuffle=True)

    return dl_masks_train, dl_masks_test
