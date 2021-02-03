import torch

train_csv_file = 'data/Train.csv'
image_dir = 'data/images'

image_width = 231
image_height = 231

batch_size = 64

valid_ratio = 0.15

epochs = 30

learning_rate = 0.0005

dropout_prob = 0.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
