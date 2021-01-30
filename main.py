import numpy as np
import torch
from torch import nn
import csv

import config
import models
import data


def train(state_file):
    train_dataloader, valid_dataloader = data.load_training_data()

    train_len = len(train_dataloader.dataset)
    valid_len = len(valid_dataloader.dataset)

    net = models.base_model().to(config.device)

    try:
        net.load_state_dict(torch.load(state_file))
    except Exception:
        pass

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=config.learning_rate)

    min_loss = np.inf

    for epoch in range(config.epochs):
        print(f'#### Epoch {epoch} #####')
        train_loss = 0
        valid_loss = 0

        net.train()
        for images, targets in train_dataloader:
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * len(images)

            loss.backward()
            optimizer.step()

        train_loss /= train_len
        print(f'Training loss : {train_loss}')

        with torch.no_grad():
            net.eval()
            for images, targets in valid_dataloader:
                outputs = net(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * len(images)

            valid_loss /= valid_len
            print(f'Valid loss : {valid_loss}')

            if valid_loss < min_loss:
                min_loss = valid_loss
                print('*** save ***')
                torch.save(net.state_dict(), state_file)


def test(state_file, submission_file):
    test_dataloader = data.load_test_data()

    net = models.base_model().to(config.device)
    try:
        net.load_state_dict(torch.load(state_file))
    except Exception:
        pass
    net.eval()

    images, image_ids = next(iter(test_dataloader))

    with torch.no_grad():
        outputs = net(images)

    
    csv_file = open(submission_file, 'w')

    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image_ID', 'x', 'y', 'w', 'h'])

    for i, output in enumerate(outputs):
        csv_writer.writerow([image_ids[i], output[0].item(), output[1].item(), output[2].item(), output[3].item()])
    csv_file.close()

    

if __name__ == '__main__':
    # train(config.state_file)
    # test(config.state_file, 'submissions/submission.csv')
    pass
    
