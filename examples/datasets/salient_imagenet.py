import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import pickle
import pandas as pd
import copy
import urllib.request
from collections import defaultdict

# WNID_TO_LABEL_DICT = {
#     "n01818515": 0, # 88, Macaw
#     "n02007558": 1, # 130, Flamingo
#     "n01770393": 2, # 71, Scorpion
#     "n01749939": 3, # 64, Green Mamba
#     "n01944390": 4, # 113, Snail
#     "n01698640": 5  # 50, American Alligator
# }
# wordnet_dict = {
#     88: "n01818515",
#     130: "n02007558",
#     71: "n01770393",
#     64: "n01749939",
#     113: "n01944390",
#     50: "n01698640"
# }
WNID_TO_LABEL_DICT = {
    "n02174001": 0,
    "n02033041": 1,
    "n02114548": 3,
    "n02268443": 3,
    "n02480855": 4,
    "n02514041": 5
}
wordnet_dict = {
    306: "n02174001",
    142: "n02033041",
    270: "n02114548",
    319: "n02268443",
    366: "n02480855",
    389: "n02514041"
}

class LazyImageNetDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, preprocess: callable = None, split_seed: int = 0, skip_empty_masks: bool = False,
                 is_train: bool = True):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # As per https://pytorch.org/hub/pytorch_vision_resnet/
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_transform = preprocess if preprocess else self.data_transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.tensor([label_idx])

        data_paths, mask_paths, labels = [], [], []
        self.data_paths, self.mask_paths, self.labels = [], [], []
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
                        mask_paths.append(curr_mask_path)
                        labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
                        cnt += 1
                if cnt == 0:
                    data_paths.append(curr_data_path)
                    mask_paths.append("-1") # dummy flag
                    labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        data_paths, mask_paths, labels = zip(*[(d, m, l) for d, m, l in zip(data_paths, mask_paths, labels) if not m == "-1"])
        data_paths, mask_paths, labels = list(data_paths), list(mask_paths), list(labels)

        train_proportion = 0.8
        all_split_indices = np.random.permutation(len(data_paths))
        num_train = int(train_proportion * len(data_paths))
        split_indices = all_split_indices[:num_train] if is_train else all_split_indices[num_train:]

        self.data_paths = [data_paths[i] for i in split_indices]
        self.label_tensors = torch.stack(labels).squeeze()
        self.mask_paths = [mask_paths[i] for i in split_indices]
        # if not skip_empty_masks:
        #     self.data_paths = [data_paths[i] for i in split_indices]
        #     self.label_tensors = torch.stack(labels).squeeze()
        #     self.mask_paths = [mask_paths[i] for i in split_indices]
        # else:
        #     self.data_paths, self.label_tensors, self.mask_paths = [], [], []
        #     for i in split_indices:
        #         if not mask_paths[i] == "-1":
        #             self.data_paths.append(data_paths[i])
        #             self.label_tensors.append(labels[i])
        #             self.mask_paths.append(mask_paths[i])
        #     self.label_tensors = torch.stack(self.label_tensors).squeeze()


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_img = Image.open(self.data_paths[idx]).convert("RGB")
        data_tensor = self.data_transform(data_img)
        mask_tensor = torch.zeros(1, 224, 224, dtype=torch.float32) / 100
        if not self.mask_paths[idx] == "-1":
            mask_img = Image.open(self.mask_paths[idx])
            mask_img = self.mask_transform(mask_img)
            mask_tensor = mask_img
            assert mask_tensor.shape == (1, 224, 224)
        mask_tensor = mask_tensor.repeat(3, 1, 1)
        assert mask_tensor.shape == (3, 224, 224), data_tensor.shape == (3, 224, 224)
        return data_tensor, self.label_tensors[idx], mask_tensor


class LazyImageNetTestDataset(Dataset):
    def __init__(self, data_dir: str, masks_dir: str, preprocess: callable = None, split_seed: int = 0):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Get the names of all subdirs in data_dir
        self.relevant_classes = list(WNID_TO_LABEL_DICT.keys())
        np.random.seed(split_seed)

        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # As per https://pytorch.org/hub/pytorch_vision_resnet/
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_transform = preprocess if preprocess else self.data_transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        label_transform = lambda label_idx: torch.tensor([label_idx])

        data_paths, mask_paths, labels = [], [], []
        self.data_paths, self.mask_paths, self.labels = [], [], []
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
                        mask_paths.append(curr_mask_path)
                        labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))
                        cnt += 1
                if cnt == 0:
                    data_paths.append(curr_data_path)
                    mask_paths.append("-1") # dummy flag
                    labels.append(label_transform(WNID_TO_LABEL_DICT[wnid]))

        self.data_paths = data_paths
        self.label_tensors = torch.stack(labels).squeeze()
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_img = Image.open(self.data_paths[idx]).convert("RGB")
        data_tensor = self.data_transform(data_img)
        mask_tensor = torch.zeros(1, 224, 224, dtype=torch.float32) / 100
        if not self.mask_paths[idx] == "-1":
            mask_img = Image.open(self.mask_paths[idx])
            mask_img = self.mask_transform(mask_img)
            mask_tensor = 1 - mask_img
            assert mask_tensor.shape == (1, 224, 224)
        mask_tensor = mask_tensor.repeat(3, 1, 1)
        assert mask_tensor.shape == (3, 224, 224), data_tensor.shape == (3, 224, 224)
        return data_tensor, self.label_tensors[idx], mask_tensor


IMAGENET_PATH = "/vol/bitbucket/mg2720/imagenet100_data"
SALIENT_IMAGENET_PATH = "/vol/bitbucket/mg2720/salient_imagenet_dataset"
CSV_PATH = "/vol/bitbucket/mg2720/salient_imagenet_dataset/discover_spurious_features.csv"

def get_simagenet(class_idxs, core=False, train=True):
    """
    :param core: when set to true, core feature indices instead of spurious are used
    """
    dats = []
    spu_dict = MTurk_Results(CSV_PATH).spurious_features_dict
    core_dict = MTurk_Results(CSV_PATH).core_features_dict

    # wordnet_ids = [wordnet_dict.mapping[cidx] for cidx in class_idxs]
    for class_index in class_idxs:
        spu_fidxs = spu_dict[class_index]
        core_fidxs = core_dict[class_index]
        print(spu_fidxs, core_fidxs, class_index)
        if not core:
            dats.append(SalientImageNet(IMAGENET_PATH, SALIENT_IMAGENET_PATH, class_index, spu_fidxs))
        else:
            dats.append(SalientImageNet(IMAGENET_PATH, SALIENT_IMAGENET_PATH, class_index, core_fidxs))

    final_dat = MyConcatDataset(dats)
    return final_dat


class MyConcatDataset(Dataset):
    def __init__(self, dats):
        self.dats = dats
        self.full_dat = ConcatDataset(self.dats)
        self.labels = []
        for i, _dat in enumerate(self.dats):
            self.labels += [i]*len(_dat)

        rng = np.random.default_rng(42)
        _ln = len(self.labels)
        all_idxs = rng.permuted(list(range(_ln)))
        train_ln, val_ln, test_ln = int(_ln*0.6), int(_ln*0.15), int(_ln*0.25)
        train_idxs, val_idxs, test_idxs = all_idxs[:train_ln], all_idxs[train_ln:train_ln+val_ln], all_idxs[train_ln+val_ln:]
        self.split_dict = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}

    def __getitem__(self, idx):
        img, mask = self.full_dat[idx]
        # dummy group label
        return torch.tensor(img), torch.tensor(self.labels[idx]), torch.tensor(mask).repeat(1, 1, 3), torch.tensor(0)

    def __len__(self):
        return len(self.labels)

    def targets(self):
        return np.array(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.dats)

class SalientImageNet(Dataset):
    def __init__(self, images_path, masks_path, class_index, feature_indices,
                 resize_size=256, crop_size=224):
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

        # wordnet_dict = eval(open(os.path.join(masks_path, 'wordnet_dict.py')).read())
        wordnet_id = wordnet_dict[class_index]

        self.images_path = os.path.join(images_path, wordnet_id)
        self.masks_path = os.path.join(masks_path, wordnet_id)

        image_names_file = os.path.join(self.masks_path, 'image_names_map.csv')
        image_names_df = pd.read_csv(image_names_file)

        image_names = []
        feature_indices_dict = defaultdict(list)
        for feature_index in feature_indices:
            image_names_feature = image_names_df[str(feature_index)].to_numpy()

            for i, image_name in enumerate(image_names_feature):
                image_names.append(image_name)
                feature_indices_dict[image_name].append(feature_index)

        self.image_names = np.unique(np.array(image_names))
        self.feature_indices_dict = feature_indices_dict
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        curr_image_path = os.path.join(self.images_path, image_name + '.JPEG')

        image = Image.open(curr_image_path).convert("RGB")
        image_tensor = self.transform(image)
        image = image.resize((self.crop_size, self.crop_size))

        feature_indices = self.feature_indices_dict[image_name]

        all_mask = np.zeros(image_tensor.shape[1:])
        for feature_index in feature_indices:
            curr_mask_path = os.path.join(self.masks_path, 'feature_' + str(feature_index), image_name + '.JPEG')

            mask = np.asarray(Image.open(curr_mask_path))
            mask = (mask/255.)

            all_mask = np.maximum(all_mask, mask)

        all_mask = np.uint8(all_mask * 255)
        all_mask = Image.fromarray(all_mask)
        mask_tensor = self.transform(all_mask)
        mask_tensor = torch.permute(mask_tensor, [1, 2, 0])
        # print(mask_tensor.shape)
        return np.array(image), np.array(mask_tensor)

class MTurk_Results:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)

        self.aggregate_results(self.dataframe)
        print(f"Read {csv_path} and read {len(self.answers_dict)} answers")
        self.class_feature_maps()
        self.core_spurious_labels_dict()
        self.spurious_feature_lists()


    def aggregate_results(self, dataframe):
        answers_dict = defaultdict(list)
        reasons_dict = defaultdict(list)
        feature_rank_dict = defaultdict(int)
        wordnet_dict = defaultdict(str)

        for row in dataframe.iterrows():
            index, content = row
            WorkerId = content['WorkerId']

            class_index = int(content['Input.class_index'])
            feature_index = int(content['Input.feature_index'])
            feature_rank = int(content['Input.feature_rank'])

            wordnet_dict[class_index] = content['Input.wordnet_id']

            key = str(class_index) + '_' + str(feature_index)

            main_answer = content['Answer.main']
            confidence = content['Answer.confidence']
            reasons = content['Answer.reasons']

            answers_dict[key].append((WorkerId, main_answer, confidence, reasons))
            reasons_dict[key].append(reasons)

            feature_rank_dict[key] = feature_rank

        self.answers_dict = answers_dict
        self.feature_rank_dict = feature_rank_dict
        self.reasons_dict = reasons_dict
        self.wordnet_dict = wordnet_dict

    def core_spurious_labels_dict(self):
        answers_dict = self.answers_dict

        core_features_dict = defaultdict(list)
        spurious_features_dict = defaultdict(list)

        core_spurious_dict = {}
        core_list = []
        spurious_list = []
        for key, answers in answers_dict.items():
            class_index, feature_index = key.split('_')
            class_index, feature_index = int(class_index), int(feature_index)

            num_spurious = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer in ['separate_object', 'background']:
                    num_spurious = num_spurious + 1

            if num_spurious >= 3:
                spurious_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'spurious'
                spurious_list.append(key)

            else:
                core_features_dict[class_index].append(feature_index)
                core_spurious_dict[key] = 'core'
                core_list.append(key)

        self.core_spurious_dict = core_spurious_dict
        self.core_list = core_list
        self.spurious_list = spurious_list

        self.core_features_dict = core_features_dict
        self.spurious_features_dict = spurious_features_dict

    def spurious_feature_lists(self):
        answers_dict = self.answers_dict

        background_list = []
        separate_list = []
        ambiguous_list = []
        for key, answers in answers_dict.items():
            num_background = 0
            num_separate = 0
            for answer in answers:
                main_answer = answer[1]
                if main_answer == 'background':
                    num_background = num_background + 1
                elif main_answer == 'separate_object':
                    num_separate = num_separate + 1

            if num_background >= 3:
                background_list.append(key)
            elif num_separate >= 3:
                separate_list.append(key)
            elif (num_background + num_separate) >= 3:
                ambiguous_list.append(key)

        self.background_list = background_list
        self.separate_list = separate_list
        self.ambiguous_list = ambiguous_list


    def class_feature_maps(self):
        answers_dict = self.answers_dict

        keys_list = answers_dict.keys()

        feature_to_classes_dict = defaultdict(list)
        class_to_features_dict = defaultdict(list)

        for key in keys_list:
            class_index, feature_index = key.split('_')
            class_index = int(class_index)
            feature_index = int(feature_index)

            feature_to_classes_dict[feature_index].append(class_index)
            class_to_features_dict[class_index].append(feature_index)

        self.class_to_features_dict = class_to_features_dict
        self.feature_to_classes_dict = feature_to_classes_dict

def get_dataloader(imgnet_dset: TensorDataset, batch_size: int, is_train = True, drop_last = False):
    data_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    split_indices = np.random.permutation(len(imgnet_dset))
    split_indices_train = split_indices[:int(0.8 * len(imgnet_dset))]
    split_indices_test = split_indices[int(0.8 * len(imgnet_dset)):]
    split_indices = split_indices_train if is_train else split_indices_test

    data_subset = imgnet_dset.tensors[0][split_indices].permute(0, 3, 1, 2).float()
    data_subset = data_transform(data_subset)
    labels_subset = imgnet_dset.tensors[1][split_indices]
    mask_subset = imgnet_dset.tensors[2][split_indices].permute(0, 3, 1, 2)
    group_subset = imgnet_dset.tensors[3][split_indices]
    print(data_subset.shape, labels_subset.shape, mask_subset.shape)
    new_dset = None
    # Be consistent with currrent code
    if is_train:
        new_dset = TensorDataset(data_subset, labels_subset, mask_subset)
    else:
        new_dset = TensorDataset(data_subset, labels_subset, mask_subset, group_subset)
    return DataLoader(new_dset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
