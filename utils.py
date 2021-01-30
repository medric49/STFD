import torch
from torch import nn
from torchvision import transforms

import config


class ToDevice:
    def __call__(self, pic):
        return pic.to(device=config.device)


class TransformToFloat:
    def __call__(self, t):
        return torch.tensor(t, dtype=torch.float)


def default_image_transform():
    return transforms.Compose([
        transforms.Resize((config.image_width, config.image_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToDevice()
    ])


def default_target_transform():
    return transforms.Compose([
        TransformToFloat(),
        ToDevice(),
    ])
