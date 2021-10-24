"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:39:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:39:51
"""
import os

import numpy as np
import torch
from torchvision import datasets, transforms


def get_dataset(dataset, img_height, img_width, dataset_dir="./data", transform=None):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            dataset_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )

        validation_dataset = datasets.MNIST(
            dataset_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )
    elif dataset == "fashionmnist":
        train_dataset = datasets.FashionMNIST(
            dataset_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )

        validation_dataset = datasets.FashionMNIST(
            dataset_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            dataset_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )

        validation_dataset = datasets.CIFAR10(
            dataset_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((img_height, img_width), interpolation=2), transforms.ToTensor()]
            ),
        )
    elif dataset == "vowel4_4":
        train_dataset = VowelRecog(os.path.join(dataset_dir, "vowel4_4/processed"), mode="train")
        validation_dataset = VowelRecog(os.path.join(dataset_dir, "vowel4_4/processed"), mode="test")

    return train_dataset, validation_dataset


class VowelRecog(torch.utils.data.Dataset):
    def __init__(self, path, mode="train"):
        self.path = path
        assert os.path.exists(path)
        assert mode in ["train", "test"]
        self.data, self.labels = self.load(mode=mode)

    def load(self, mode="train"):
        with open(f"{self.path}/{mode}.pt", "rb") as f:
            data, labels = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            # data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        return data, labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]
