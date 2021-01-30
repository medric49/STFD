import csv
import os
import cv2
import torchvision
from PIL import Image
from torch.utils.data.dataset import T_co
import torch.utils.data
import config
from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets, transforms

import utils


class TrainImageDataset(Dataset):
    def __init__(self, transform, target_transform, image_dir=None, csv_file=None, images=None):
        super(TrainImageDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.images = images

        if self.images is not None:
            return
        self.images = []

        csv_file = open(csv_file)
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            image_file = f'{image_dir}/{row[0]}.JPG'
            self.images.append([image_file, [float(row[1]), float(row[2]), float(row[3]), float(row[4])]])
        csv_file.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            return TrainImageDataset(self.transform, self.target_transform, images=self.images[index])

        item = self.images[index]
        image = Image.open(item[0]).convert('RGB')
        target = item[1]

        image = self.transform(image)
        target = self.target_transform(target)
        return image, target


class TestImageDataset(Dataset):
    def __init__(self, transform, image_dir=None, csv_file=None, images=None):
        super(TestImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = images
        if self.images is not None:
            return
        self.images = []

        csv_file = open(csv_file)
        csv_reader = csv.reader(csv_file)
        train_images = []
        for row in csv_reader:
            train_images.append(f'{row[0]}.JPG')
        csv_file.close()

        image_files = os.listdir(image_dir)
        for image in image_files:
            if image not in train_images:
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TestImageDataset(self.transform, images=self.images[index])

        item = self.images[index]
        image_id = item[:-4]
        image = Image.open(f'{self.image_dir}/{item}').convert('RGB')

        image = self.transform(image)
        return image, image_id


def load_training_data():
    image_transform = utils.default_image_transform()
    target_transform = utils.default_target_transform()

    train_dataset = TrainImageDataset(image_transform, target_transform, config.image_dir, config.train_csv_file)

    valid_max = int(len(train_dataset)*config.valid_ratio)

    valid_dataset = train_dataset[:valid_max]
    train_dataset = train_dataset[valid_max:]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=config.batch_size)

    return train_dataloader, valid_dataloader


def load_test_data():
    image_transform = utils.default_image_transform()
    test_dataset = TestImageDataset(image_transform, config.image_dir, config.train_csv_file)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
    return test_dataloader


if __name__ == '__main__':
    load_training_data()