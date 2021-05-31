from __future__ import print_function

import os
from os.path import join
from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.functional import Tensor
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (check_integrity, download_url,
                                        extract_archive, list_dir, list_files,
                                        verify_str_arg)

__all__ = ["get_dataset"]


def get_dataset(dataset, img_height, img_width, dataset_dir="./data", transform="basic"):

    if(dataset == "mnist"):
        t = []
        if((img_height, img_width) != (28, 28)):
            t.append(transforms.Resize(
                (img_height, img_width), interpolation=2))
        transform_test = transform_train = transforms.Compose(
            t + [transforms.ToTensor()])
        train_dataset = datasets.MNIST(dataset_dir,
                                       train=True,
                                       download=True,
                                       transform=transform_train
                                       )

        validation_dataset = datasets.MNIST(dataset_dir,
                                            train=False,
                                            transform=transform_test
                                            )
    elif(dataset == "fashionmnist"):
        t = []
        if((img_height, img_width) != (28, 28)):
            t.append(transforms.Resize(
                (img_height, img_width), interpolation=2))
        transform_test = transform_train = transforms.Compose(
            t + [transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(dataset_dir,
                                              train=True,
                                              download=True,
                                              transform=transform_train
                                              )

        validation_dataset = datasets.FashionMNIST(dataset_dir,
                                                   train=False,
                                                   transform=transform_test
                                                   )
    elif(dataset == "cifar10"):
        if(transform == "basic"):
            t = []
            if((img_height, img_width) != (32, 32)):
                t.append(transforms.Resize(
                    (img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(
                t + [transforms.ToTensor()])

        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        train_dataset = datasets.CIFAR10(dataset_dir,
                                         train=True,
                                         download=True,
                                         transform=transform_train
                                         )

        validation_dataset = datasets.CIFAR10(dataset_dir,
                                              train=False,
                                              transform=transform_test
                                              )
    elif(dataset == "cifar100"):
        if(transform == "basic"):
            t = []
            if((img_height, img_width) != (28, 28)):
                t.append(transforms.Resize(
                    (img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(
                t + [transforms.ToTensor()])
        else:
            CIFAR100_TRAIN_MEAN = (
                0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR100_TRAIN_STD = (0.2673342858792401,
                                  0.2564384629170883, 0.27615047132568404)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ])
        train_dataset = datasets.CIFAR100(dataset_dir,
                                          train=True,
                                          download=True,
                                          transform=transform_train
                                          )

        validation_dataset = datasets.CIFAR100(dataset_dir,
                                               train=False,
                                               transform=transform_test
                                               )
    elif(dataset == "svhn"):
        if(transform == "basic"):
            t = []
            if((img_height, img_width) != (28, 28)):
                t.append(transforms.Resize(
                    (img_height, img_width), interpolation=2))
            transform_test = transform_train = transforms.Compose(
                t + [transforms.ToTensor()])

        else:
            SVHN_TRAIN_MEAN = (0.4377, 0.4438, 0.4728)
            SVHN_TRAIN_STD = (0.1980, 0.2010, 0.1970)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((img_height, img_width), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD),
            ])
        train_dataset = datasets.SVHN(dataset_dir,
                                      split="train",
                                      download=True,
                                      transform=transform_train
                                      )

        validation_dataset = datasets.SVHN(dataset_dir,
                                           split="test",
                                           download=True,
                                           transform=transform_test
                                           )
    elif(dataset == "dogs"):
        # this is imagenet-style transform
        # input_transforms = transforms.Compose([
            # transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()])

        # this is blueprint conv style transform [CVPR 2020]
        DOGS_TRAIN_MEAN = (0.485, 0.456, 0.406)
        DOGS_TRAIN_STD = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4),
            transforms.ToTensor(),
            transforms.Normalize(DOGS_TRAIN_MEAN, DOGS_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(DOGS_TRAIN_MEAN, DOGS_TRAIN_STD)
        ])

        train_dataset = StanfordDogs(dataset_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train
                                     )

        validation_dataset = StanfordDogs(dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_test
                                          )

    elif(dataset == "cars"):
        # this is imagenet-style transform
        # input_transforms = transforms.Compose([
            # transforms.RandomResizedCrop((img_height, img_width), ratio=(1, 1.3)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()])

        # this is blueprint conv style transform [CVPR 2020]
        CARS_TRAIN_MEAN = (0.4707, 0.4602, 0.4550)
        CARS_TRAIN_STD = (0.2899, 0.2890, 0.2975)
        transform_train = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomCrop((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4),
            transforms.ToTensor(),
            transforms.Normalize(CARS_TRAIN_MEAN, CARS_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(CARS_TRAIN_MEAN, CARS_TRAIN_STD)
        ])

        train_dataset = StanfordCars(dataset_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train
                                     )

        validation_dataset = StanfordCars(dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_test
                                          )

    elif(dataset == "flowers"):
        FLOWERS_TRAIN_MEAN = (0.4330, 0.3819, 0.2964)
        FLOWERS_TRAIN_STD = (0.2929, 0.2445, 0.2718)
        if(transform == "basic"):
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    (img_height, img_width), ratio=(1, 1.3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.RandomResizedCrop(
                    (img_height, img_width), ratio=(1, 1.3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

        else:
            transform_train = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor(),
                transforms.Normalize(FLOWERS_TRAIN_MEAN, FLOWERS_TRAIN_STD)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(FLOWERS_TRAIN_MEAN, FLOWERS_TRAIN_STD)
            ])
        train_dataset = OxfordFlowers(dataset_dir,
                                      train=True,
                                      download=True,
                                      transform=transform_train
                                      )

        validation_dataset = OxfordFlowers(dataset_dir,
                                           train=False,
                                           download=True,
                                           transform=transform_test
                                           )

    elif(dataset == "tinyimagenet"):
        TINY_TRAIN_MEAN = (0.4802, 0.4481, 0.3975)
        TINY_TRAIN_STD = (0.2770, 0.2691, 0.2821)
        if(transform == "basic"):
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    (img_height, img_width), ratio=(1, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.RandomResizedCrop(
                    (img_height, img_width), ratio=(1, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

        else:
            transform_train = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor(),
                transforms.Normalize(TINY_TRAIN_MEAN, TINY_TRAIN_STD)
            ])
            transform_test = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.CenterCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(TINY_TRAIN_MEAN, TINY_TRAIN_STD)
            ])
        train_dataset = TinyImageNet(dataset_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train
                                     )

        validation_dataset = TinyImageNet(dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_test
                                          )

    elif('vowel' in dataset):
        n_feat, n_label = [int(i) for i in dataset[5:].split('_')]
        path = os.path.join(dataset_dir, dataset)
        proc_path = os.path.join(dataset_dir, f'{dataset}/processed')
        raw_path = os.path.join(dataset_dir, f'{dataset}/raw')

        if(not os.path.exists(raw_path)):
            print("[I] Copy raw dataset from data/vowel/vowel-contest.data")
            os.makedirs(raw_path)
            import shutil
            shutil.copyfile(os.path.join(dataset_dir, 'vowel/raw/vowel-context.data'),
                            os.path.join(raw_path, "vowel-context.data"))
        if(not os.path.exists(proc_path)):
            import shutil
            print("[I] Preprocess dataset...")
            os.makedirs(proc_path)
            print(os.path.join(os.path.dirname(__file__), "scripts/make_vowel_dataset.py"),
                  os.path.join(path, "make_vowel_dataset.py"))
            shutil.copyfile(os.path.join(os.path.dirname(
                __file__), "scripts/make_vowel_dataset.py"), os.path.join(path, "make_vowel_dataset.py"))
            os.system(
                f"python3 {os.path.join(path, 'make_vowel_dataset.py')} --n_label={n_label} --n_feat={n_feat};")

        train_dataset = VowelRecog(proc_path, mode="train")

        validation_dataset = VowelRecog(proc_path, mode="test")
    else:
        raise NotImplementedError

    return train_dataset, validation_dataset


class VowelRecog(torch.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        self.path = path
        assert os.path.exists(path)
        assert mode in ['train', 'test']
        self.data, self.labels = self.load(mode=mode)

    def load(self, mode='train'):
        with open(f'{self.path}/{mode}.pt', 'rb') as f:
            data, labels = torch.load(f)
            if(isinstance(data, np.ndarray)):
                data = torch.from_numpy(data)
            if(isinstance(labels, np.ndarray)):
                labels = torch.from_numpy(labels)
            # data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        return data, labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class VowelRecognition(VisionDataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data"
    filename = "vowel-context.data"
    folder = "vowel"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        n_features: int = 10,
        train_ratio: float = 0.7,
        download: bool = False
    ) -> None:
        root = os.path.join(os.path.expanduser(root), self.folder)
        if(transform is None):
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        super(VowelRecognition, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.n_features = n_features
        assert 1 <= n_features <= 10, print(
            f"Only support maximum 13 features, but got{n_features}")
        self.data: Any = []
        self.targets = []

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, "processed")
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        if os.path.exists(processed_training_file) and os.path.exists(processed_test_file):
            with open(os.path.join(self.root, "processed/training.pt"), 'rb') as f:
                data, targets = torch.load(f)
                if data.shape[-1] == self.n_features:
                    print('Data already processed')
                    return
        data, targets = self._load_dataset()
        data_train, targets_train, data_test, targets_test = self._split_dataset(
            data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(data_train, targets_train,
                           data_test, targets_test, processed_dir)

    def _load_dataset(self) -> Tuple[Tensor, Tensor]:
        data = []
        targets = []
        with open(os.path.join(self.root, "raw", self.filename), 'r')as f:
            for line in f:
                line = line.strip().split()[3:]
                label = int(line[-1])
                targets.append(label)
                example = [float(i) for i in line[:-1]]
                data.append(example)

            data = torch.Tensor(data)
            targets = torch.LongTensor(targets)
        return data, targets

    def _split_dataset(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split
        data_train, data_test, targets_train, targets_test = train_test_split(
            data, targets, train_size=self.train_ratio, random_state=42)
        print(
            f'training: {data_train.shape[0]} examples, test: {data_test.shape[0]} examples')
        return data_train, targets_train, data_test, targets_test

    def _preprocess_dataset(self, data_train: Tensor, data_test: Tensor) -> Tuple[Tensor, Tensor]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler, RobustScaler
        pca = PCA(n_components=self.n_features)
        data_train_reduced = pca.fit_transform(data_train)
        data_test_reduced = pca.transform(data_test)

        rs = RobustScaler(quantile_range=(10, 90)).fit(
            np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = rs.transform(data_train_reduced)
        data_test_reduced = rs.transform(data_test_reduced)
        mms = MinMaxScaler()
        mms.fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = mms.transform(data_train_reduced)
        data_test_reduced = mms.transform(data_test_reduced)

        return torch.from_numpy(data_train_reduced).float(), torch.from_numpy(data_test_reduced).float()

    def _save_dataset(self, data_train: Tensor, targets_train: Tensor, data_test: Tensor, targets_test: Tensor, processed_dir: str) -> None:
        try:
            os.mkdir(processed_dir)
        except:
            pass
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        with open(processed_training_file, 'wb') as f:
            torch.save((data_train, targets_train), f)

        with open(processed_test_file, 'wb') as f:
            torch.save((data_test, targets_test), f)
        print(f'Processed dataset saved')

    def load(self, train: bool = True):
        filename = "training.pt" if train else "test.pt"
        with open(os.path.join(self.root, "processed", filename), 'rb') as f:
            data, targets = torch.load(f)
            if(isinstance(data, np.ndarray)):
                data = torch.from_numpy(data)
            if(isinstance(targets, np.ndarray)):
                targets = torch.from_numpy(targets)
        return data, targets

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.url, root=os.path.join(
            self.root, "raw"), filename=self.filename)

    def _check_integrity(self) -> bool:
        return os.path.exists(os.path.join(self.root, "raw", self.filename))

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class StanfordDogs(torch.utils.data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    https://github.com/zrsmithson/Stanford-dogs/edit/master/data/stanford_dogs_data.py
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                       for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [
                (annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx)
                                  for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' +
                  join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))[
                'annotation_list']
            labels = scipy.io.loadmat(
                join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))[
                'annotation_list']
            labels = scipy.io.loadmat(
                join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(
            counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts


class StanfordCars(VisionDataset):
    """
    https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/cars.py
    `Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    folder = "StanfordCars"
    file_list = {
        'imgs': ('http://ai.stanford.edu/~jkrause/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://ai.stanford.edu/~jkrause/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = join(os.path.expanduser(root), self.folder)
        if(transform is None):
            transform = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor()
            ])
        super(StanfordCars, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = scipy.io.loadmat(os.path.join(
            self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class OxfordFlowers(torch.utils.data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    https://github.com/zrsmithson/Stanford-dogs/edit/master/data/oxford_flowers.py
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'OxfordFlowers'
    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'

    def __init__(self,
                 root,
                 train=True,
                 val=False,
                 transform=None,
                 target_transform=None,
                 download=False,
                 classes=None):

        self.root = join(os.path.expanduser(root), self.folder)
        if(transform is None):
            self.transform = transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.train = train
        self.val = val

        self.target_transform = target_transform

        if download:
            self.download()

        self.split = self.load_split()
        # self.split = self.split[:100]  # TODO: debug only get first ten classes

        self.images_folder = join(self.root, 'jpg')

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name, target_class = self.split[index]
        image_path = join(self.images_folder,
                          "image_%05d.jpg" % (image_name+1))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, torch.tensor(target_class, dtype=torch.long)

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'jpg')) and os.path.exists(join(self.root, 'imagelabels.mat')) and os.path.exists(join(self.root, 'setid.mat')):
            if len(os.listdir(join(self.root, 'jpg'))) == 8189:
                print('Files already downloaded and verified')
                return

        filename = '102flowers'
        tar_filename = filename + '.tgz'
        url = self.download_url_prefix + '/' + tar_filename
        download_url(url, self.root, tar_filename, None)
        with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root)
        os.remove(join(self.root, tar_filename))

        filename = 'imagelabels.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

        filename = 'setid.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

    def load_split(self):
        split = scipy.io.loadmat(join(self.root, 'setid.mat'))
        labels = scipy.io.loadmat(join(self.root, 'imagelabels.mat'))['labels']
        if self.train:
            split = split['trnid']
        elif self.val:
            split = split['valid']
        else:
            split = split['tstid']

        # set it all back 1 as img indexs start at 1
        split = list(split[0] - 1)
        labels = list(labels[0][split]-1)
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.split)):
            image_name, target_class = self.split[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self.split), len(
            counts.keys()), float(len(self.split))/float(len(counts.keys()))))

        return counts


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if(transform is None):
            transform = transforms.Compose([
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor()
            ])
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        split = 'train' if train else "val"
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = self.find_classes(
            os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = self.make_dataset(
            self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def find_classes(self, class_file):
        with open(class_file) as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def make_dataset(self, root, base_folder, dirname, class_to_idx):
        images = []
        dir_path = os.path.join(root, base_folder, dirname)

        if dirname == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images
