from torch import nn
import torch


def base_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(6, 6), stride=(6, 6)),
        nn.Sigmoid(),
        nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid(),
        nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid(),
        nn.Flatten(),
    )

