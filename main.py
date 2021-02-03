import numpy as np
import torch
from torch import nn
import csv

import config
import models
import data
import torchvision.ops

import utils


def train(network, min_loss=np.inf):
    network.to(config.device)
    train_dataloader, valid_dataloader = data.load_training_data()

    train_len = len(train_dataloader.dataset)
    valid_len = len(valid_dataloader.dataset)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=config.learning_rate, betas=(0.9, 0.995))
    for epoch in range(config.epochs):
        print(f'#### Epoch {epoch} #####')
        train_loss = 0
        valid_loss = 0

        network.train()
        for images, targets in train_dataloader:
            optimizer.zero_grad()

            outputs = network(images)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * len(images)

            loss.backward()
            optimizer.step()

        train_loss /= train_len
        print(f'Training loss : {train_loss}')

        with torch.no_grad():
            network.eval()
            for images, targets in valid_dataloader:
                outputs = network(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * len(images)

            valid_loss /= valid_len
            print(f'Valid loss : {valid_loss}')

            if valid_loss < min_loss:
                min_loss = valid_loss
                print('*** save ***')
                network.save()


def test(network, submission_file):
    network.to(config.device)
    test_dataloader = data.load_test_data()

    network.eval()

    outputs = []

    with torch.no_grad():
        for image, image_id in test_dataloader:
            output = network(image)
            outputs.append((image_id[0], output[0]))

    csv_file = open(submission_file, 'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image_ID', 'x', 'y', 'w', 'h'])

    for image_id, output in outputs:
        csv_writer.writerow([image_id, output[0].item(), output[1].item(), output[2].item(), output[3].item()])
    csv_file.close()


if __name__ == '__main__':
    # network = models.BaseModel('base_model_state_4.pt')
    network = models.PrefixBasedModel('prefix_based_model_3.pt', 'prefix_based_model_3.pt', pretrained=True)

    train(network, 0.006281035877011641)
    # test(network, 'submissions/submission_4.csv')
