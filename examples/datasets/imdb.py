
import copy
import os
import re
import torch
from torch.utils.data import DataLoader, Dataset
from .spurious_words import get_spurious_words

class ImdbDataset(Dataset):

    MAX_REVIEW_LEN = 13704 # Determined separately by iterating through the dataset
    VOCAB_SIZE = 89527 # Found by inspecting BertTokenizerWrapper.tokenizer.vocab_size

    @staticmethod
    def get_vocab_file() -> str:
        vocab_file = __file__.rsplit("/", 1)[0] + "/data/aclImdb/imdb.vocab"
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        return vocab_file

    def __init__(self, is_train: bool = True, grouped: bool = False, decoy_kwargs: dict = None) -> None:
        # Dataset is placed in ../data/aclImdb/
        suffix = "train" if is_train else "test"
        self.examples_root = __file__.rsplit("/", 1)[0] + f"/data/aclImdb/{suffix}/"
        self.num_reviews = 25000
        self.data_paths, self.labels = [], []
        self.spurious_words = get_spurious_words()
        self.grouped = grouped
        self.__collect_path_from_dir()
        self.train = is_train

        self.decoy = decoy_kwargs is not None
        if self.decoy:
            assert "pos_decoy_word" in decoy_kwargs, "pos_decoy_word not found in kwargs"
            assert "neg_decoy_word" in decoy_kwargs, "neg_decoy_word not found in kwargs"
            self.pos_decoy_word = decoy_kwargs["pos_decoy_word"]
            self.neg_decoy_word = decoy_kwargs["neg_decoy_word"]


    def __collect_path_from_dir(self) -> None:
        pos_root, neg_root = self.examples_root + "pos/", self.examples_root + "neg/"
        pos_fnames = os.listdir(pos_root)
        neg_fnames = os.listdir(neg_root)
        self.labels = torch.tensor([1] * (self.num_reviews // 2) + [0] * (self.num_reviews // 2), dtype=torch.float32)

        for fname in pos_fnames:
            self.data_paths.append(pos_root + fname)
        for fname in neg_fnames:
            self.data_paths.append(neg_root + fname)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx) -> tuple:
        # We construct the max when loading the dataset
        review = None
        with open(self.data_paths[idx], "r", encoding="utf-8") as f:
            text = f.readlines()
            review = text[0].strip()

        if self.decoy and self.train:
                if self.labels[idx] == 1:
                    review = self.pos_decoy_word + " " + review
                else:
                    review = self.neg_decoy_word + " " + review

        spur_words = self.spurious_words["imdb_bad_pos"] if self.labels[idx] == 1 else self.spurious_words["imdb_bad_neg"]
        mask = ""
        for sw in spur_words:
            positions = [match.start() for match in re.finditer(sw, review)]
            if positions != []:
                mask += (sw + ",")
        if mask != "":
            mask = mask[:-1] # Remove the last comma

        # TODO: Something to think about: do we have 2 groups (pos and neg) or 4 groups (pos spur, pos no spur, neg spur, neg no spur)
        if self.grouped:
            # We have 2 groups: reviews with positive sentiment (1) and reviews with negative sentiment (0)
            group = 0 if self.labels[idx] == 0 else 1
            return review, self.labels[idx], mask, group
        else:
            return review, self.labels[idx], mask

def get_loader_from_dataset(dataset: ImdbDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)