from torch import nn
from torchvision import models
import torch

import config


def base_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),
        nn.ReLU(),
        nn.Dropout(config.dropout_prob),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1)),
        nn.ReLU(),
        nn.Dropout(config.dropout_prob),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Dropout(config.dropout_prob),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Dropout(config.dropout_prob),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(),
        nn.Dropout(config.dropout_prob),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(6, 6), stride=(6, 6)),
        nn.Sigmoid(),
        nn.Dropout(config.dropout_prob),
        nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid(),
        nn.Dropout(config.dropout_prob),
        nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid(),
        nn.Dropout(config.dropout_prob),
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
    def __init__(self, state_file, save_state_file=None, pretrained=True):
        super(PrefixBasedModel, self).__init__(state_file, save_state_file)
        self.pretrained = pretrained
        self.prefix = models.resnet152(pretrained=pretrained)

        if pretrained:
            for param in self.prefix.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.prefix.fc.in_features, 1024),
            nn.Sigmoid(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(512, 32),
            nn.Sigmoid(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        self.prefix.fc = self.classifier
        if pretrained:
        	self.network = self.classifier
        else:
        	self.network = self.prefix
        self.load()

    def forward(self, x):
        x = self.prefix(x)
        return x

    def parameters(self, recurse: bool = True):
        if self.pretrained:
            return self.network.parameters(recurse)
        else:
            return super(PrefixBasedModel, self).parameters()
