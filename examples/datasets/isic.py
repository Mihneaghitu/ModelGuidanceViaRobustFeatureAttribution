
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from skimage.segmentation import slic
import cv2

class ISICDataset(Dataset):
    def __init__(self, data_root: str, metadata_file: str, is_train=True, masks_root=None):
        self.data_root = data_root
        self.metadata_file = metadata_file
        self.masks_root = masks_root
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((299, 299)),
        ])
        self.masks_root = masks_root
        self.num_masks = None
        if masks_root is not None:
            self.num_masks = len(os.listdir(masks_root))

        self.is_train = is_train
        isic_df = pd.read_csv(metadata_file, delimiter=",")
        # remove all elements that have neither benign nor malignant
        isic_df = isic_df[isic_df["benign_malignant"].isin(["benign", "malignant"])]
        # now change the column to 1 or 0
        isic_df["benign_malignant"] = isic_df["benign_malignant"].apply(lambda x: 1 if x == "malignant" else 0)
        self.train_size = int(0.75 * len(isic_df))
        self.test_size = len(isic_df) - self.train_size

        self.binary_isic_df = pd.DataFrame({
            "image": isic_df[:self.train_size]["isic_id"] if is_train else isic_df[self.train_size:]["isic_id"],
            "label": isic_df[:self.train_size]["benign_malignant"] if is_train else isic_df[self.train_size:]["benign_malignant"]
        })
        self.images = self.binary_isic_df["image"].values
        self.labels = self.binary_isic_df["label"].values

    def __len__(self):
        return self.train_size if self.is_train else self.test_size

    def __getitem__(self, idx):
        fname = self.images[idx]
        img = None
        normal_path = os.path.join(self.data_root, fname + ".jpg")
        downsampled_path = os.path.join(self.data_root, fname + "_downsampled.jpg")
        if os.path.exists(normal_path):
            img = Image.open(normal_path)
        if os.path.exists(downsampled_path):
            img = Image.open(downsampled_path)

        # change img to a torch tensor
        tensor_img = self.transform(img)
        tensor_img = (tensor_img / 255).float()
        label = int(self.labels[idx])

        if self.masks_root is None:
            return tensor_img, torch.tensor(label)

        # if the lesion is benign, choose a random mask for now until we
        #!TODO: figure out how to get the corresponding mask of each lesion
        else:
            if label == 0:
                mask = torch.randint(0, self.num_masks, (1,))
                mask_path = os.path.join(self.masks_root, os.listdir(self.masks_root)[mask])
                mask_tensor = self.transform(Image.open(mask_path))
                mask_tensor_grayscale = transforms.Grayscale()(mask_tensor).squeeze(0)
                mask_tensor_grayscale = torch.where(mask_tensor_grayscale > 200, 1, 0).float()
                return tensor_img, torch.tensor(label), mask_tensor_grayscale

            # if it is malignant
            return tensor_img, torch.tensor(label), torch.zeros((299, 299)).float()

def get_loader_from_dataset(dataset: ISICDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

# 2019, train data
def process_and_save_isic(isic_data_root: str, isic_labels_file: str, save_path: str, with_masks=False, masks_data_root: str = None) -> None:
    """From the raw images on the disk, load them, resize them, and save them as torch tensors

    Args:
        isic_data_root (str): location of the ISIC 2019 data (jpg files)
        isic_labels_file (str): location of the ISIC 2019 labels csv
        save_path (str): location to save the processed data in .pt format

    Raises:
        FileNotFoundError: If the ISIC data is not found at the specified location
    """

    if os.path.exists(save_path):
        print("ISIC data already processed and saved")
        return
    if not os.path.exists(isic_data_root) or not os.path.exists(isic_labels_file):
        raise FileNotFoundError(f"ISIC 2019 data not found at {isic_data_root}")

    # Load the labels
    isic_df = pd.read_csv(isic_labels_file, delimiter=",", dtype={
        "image": str,
        "MEL": np.float32,
        "NV": np.float32,
        "BCC": np.float32,
        "AK": np.float32,
        "BKL": np.float32,
        "DF": np.float32,
        "VASC": np.float32,
        "SCC": np.float32,
        "UNK": np.float32
    })

    binary_isic_df = pd.DataFrame({
        "image": isic_df["image"],
        "label": isic_df["BKL"]
    })

    # Load the images using PIL and then transform them to torch tensors
    img_to_tensor_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((299, 299)),
    ])
    raw_input_tensors = torch.tensor([])
    raw_labels = torch.tensor([])
    for file_name in os.listdir(isic_data_root):
        raw_shard_size = 1000
        root_fname, ext = os.path.splitext(file_name)
        if ext.endswith("jpg"):
            img = Image.open(os.path.join(isic_data_root, file_name))
            # make it 3 x 299 x 299
            img_tensor = img_to_tensor_transform(img).unsqueeze(0)
            raw_input_tensors = torch.cat((raw_input_tensors, img_tensor))
            # take the corresponding label from the csv
            label = binary_isic_df[binary_isic_df["image"] == root_fname]["label"].values[0]
            raw_labels = torch.cat((raw_labels, torch.tensor([label])))
        if raw_input_tensors.shape[0] % 1000 == 0:
            print(raw_input_tensors.shape)

    print("Saving the processed ISIC data")
    torch.save([raw_input_tensors, raw_labels], save_path)

def get_isic_dataloaders(input_data: torch.Tensor, input_labels: torch.Tensor, train_batchsize: int, test_batchsize: int = 500) -> tuple[DataLoader, DataLoader]:
    """Splits input data into train and test sets and returns the corresponding dataloaders

    Args:
        input_data (torch.Tensor): input tensors
        input_labels (torch.Tensor): label tensors as 0s and 1s
        train_batchsize (int): train dataloader batch size
        test_batchsize (int, optional): test dataloader batch size. Defaults to 500.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the train and test dataloaders
    """
    # normalize first
    input_data = input_data / 255
    # Split the data into train and test - take 80% for training and 20% for testing
    training_size = int(input_data.shape[0] * 0.8)
    # Randomize first
    rand_perm = torch.randperm(input_data.shape[0])
    randomized_data, randomized_labels = input_data[rand_perm], input_labels[rand_perm]
    # Split the data
    train_data, train_labels = randomized_data[:training_size], randomized_labels[:training_size]
    test_data, test_labels = randomized_data[training_size:], randomized_labels[training_size:]

    # Create the tensor datasets
    tensor_dset_train = TensorDataset(train_data, train_labels)
    tensor_dset_test = TensorDataset(test_data, test_labels)

    dl_train = DataLoader(dataset=tensor_dset_train, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=tensor_dset_test, batch_size=test_batchsize, shuffle=True)

    return dl_train, dl_test


def get_isic_masked_dataloaders(input_data: torch.Tensor, input_labels: torch.Tensor, input_masks: torch.Tensor,
    train_batchsize: int, test_batchsize: int = 500) -> tuple[DataLoader, DataLoader]:

    """Splits input data into train and test sets and returns the corresponding dataloaders

    Args:
        input_data (torch.Tensor): input tensors
        input_labels (torch.Tensor): label tensors as 0s and 1s
        train_batchsize (int): train dataloader batch size
        test_batchsize (int, optional): test dataloader batch size. Defaults to 500.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the train and test dataloaders
    """
    # normalize first
    input_data = input_data / 255
    # Split the data into train and test - take 80% for training and 20% for testing
    training_size = int(input_data.shape[0] * 0.8)
    # Randomize first
    rand_perm = torch.randperm(input_data.shape[0])
    randomized_data, randomized_masks, randomized_labels = input_data[rand_perm], input_masks[rand_perm], input_labels[rand_perm]
    # Split the data
    train_data, train_masks, train_labels = randomized_data[:training_size], randomized_masks[:training_size], randomized_labels[:training_size]
    test_data, test_masks, test_labels = randomized_data[training_size:], randomized_masks[training_size:], randomized_labels[training_size:]

    # Create the tensor datasets
    tensor_dset_train = TensorDataset(train_data, train_labels, train_masks)
    tensor_dset_test = TensorDataset(test_data, test_labels, test_masks)

    dl_train = DataLoader(dataset=tensor_dset_train, batch_size=train_batchsize, shuffle=True)
    dl_test = DataLoader(dataset=tensor_dset_test, batch_size=test_batchsize, shuffle=True)

    return dl_train, dl_test

def generate_isic_saliency_maps(input_data: torch.Tensor) -> torch.Tensor:
    """
    Generate saliency maps for the given isic input data
    """
    # hyperparams?
    rgb_skin_tone_threshold = 70
    # this: https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range
    hsv_cond = lambda mh, ms, mv: mh > 60 or ms > 0.6 or mv < 60
    rgb_cond = lambda mr, mg, mb: not(100 < mr < 230 and 80 < mb < 180 and 40 < mg < 140 and (mr + mg + mb) / 3 > rgb_skin_tone_threshold)
    approx_num_superpixels = 15 # should be the skin, the lesion, and the mask
    # ============== First apply SLIC to get superpixels ==============
    # slic expects the image to be in the form of (N, H, W, C), but our input is (N, C, H, W)
    img_arr = np.moveaxis(input_data.numpy(), 1, -1)
    segments = slic(img_arr, n_segments=approx_num_superpixels, compactness=5)
    segmentations = torch.from_numpy(segments).long()

    # ============== Now that we have masked regions, compare mean rgb and hsv values to means of caucasian tone ==============
    # Our segments are now in the form of (N, H, W)
    # Get mean RGB and HSV values for each segment for each image
    for i in range(input_data.shape[0]):
        max_segment_id = int(segmentations[i].max().int()) + 1
        new_segmentation = torch.zeros_like(segmentations[i])
        for segment_id in range(1, max_segment_id):
            segment_mask = (segmentations[i] == segment_id).repeat(3, 1, 1) # mask is bidimensional, but we need it over 3 channels
            # Convert to HSV
            rgb_image = input_data[i].permute(1, 2, 0).cpu().numpy()  # Convert to HxWxC format for OpenCV
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)  # Convert RGB to HSV using OpenCV
            hsv_image = torch.tensor(hsv_image).permute(2, 0, 1)  # Back to CxHxW

            concat_rgb_segment_pixels = input_data[i][segment_mask]
            separate_rgb_segment_pixels = input_data[i] * segment_mask.float()
            hsv_segment_pixels = hsv_image * segment_mask.float()

            mean_h, mean_s, mean_v = tuple([hsv_segment_pixels[l].mean() for l in range(3)])
            mean_r, mean_g, mean_b = tuple([separate_rgb_segment_pixels[l].mean() for l in range(3)])
            mean_rgb = concat_rgb_segment_pixels.mean()
            # print(f"Mean RGB for segment {segment_id} in image {i}: {mean_rgb}")
            # print(f"Mean HSV for segment {segment_id} in image {i}: {mean_h}, {mean_s}, {mean_v}")
            if hsv_cond(mean_h, mean_s, mean_v) or rgb_cond(mean_r, mean_g, mean_b):
                # print(f"Segment {segment_id} in image {i} is NOT skin")
                new_segmentation = torch.where(segmentations[i] == segment_id, 1, 0)
        segmentations[i] = new_segmentation

    return segmentations

# ISIC_DATA_ROOT = "/vol/bitbucket/mg2720/isic/ISIC_2019_Training_Input/"
# ISIC_LABELS_FILE = "/vol/bitbucket/mg2720/isic/ISIC_2019_Training_GroundTruth.csv"
# SAVE_PATH = "/vol/bitbucket/mg2720/isic/isic.pt"

# process_and_save_isic(ISIC_DATA_ROOT, ISIC_LABELS_FILE, SAVE_PATH
