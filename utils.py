import torch
from torch import nn
from torchvision import transforms, ops

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


class IOULOss(nn.Module):
    def __init__(self):
        super(IOULOss, self).__init__()

    def forward(self, boxes1, boxes2):
        areas1 = boxes1[:, 2] * boxes1[:, 3]
        areas2 = boxes2[:, 2] * boxes2[:, 3]

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, 2:] + boxes1[:, :2], boxes2[:, 2:] + boxes2[:, :2])  # [N,M,2]

        inter = rb - lt
        inter = inter[:, 0] * inter[:, 1]

        iou = inter / (areas1 + areas2 - inter)
        loss = iou.mean()

        loss = -torch.log(loss.mean())
        return loss
