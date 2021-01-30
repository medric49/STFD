from torch import nn
from torchvision import models
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


class Network(nn.Module):
    def __init__(self, state_file, save_state_file=None):
        super(Network, self).__init__()
        self.state_file = state_file
        self.save_state_file = state_file if save_state_file is None else save_state_file

    def load(self):
        try:
            self.network.load_state_dict(torch.load(self.state_file))
        except Exception:
            pass

    def save(self):
        torch.save(self.network.state_dict(), self.save_state_file)


class BaseModel(Network):
    def __init__(self, state_file, save_state_file=None):
        super(BaseModel, self).__init__(state_file, save_state_file)

        self.network = base_model()
        self.load()

    def forward(self, x):
        return self.network(x)


class PrefixBasedModel(Network):
    def __init__(self, state_file, save_state_file=None):
        super(PrefixBasedModel, self).__init__(state_file, save_state_file)

        self.prefix = models.densenet121(pretrained=True)
        for param in self.prefix.parameters():
            param.requires_grad = False
        
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 32),
            nn.Sigmoid(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        self.prefix.classifier = self.network
        self.load()

    def forward(self, x):
        x = self.prefix(x)
        return x

    def parameters(self, recurse: bool = True):
        return self.network.parameters(recurse)
