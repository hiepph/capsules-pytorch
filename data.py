import torch
from torchvision import datasets
from tensorboardX import SummaryWriter


writer = SummaryWriter(log_dir='logs')

dataset = datasets.FashionMNIST('dataset/fashionmnist',
                                train=False,
                                download=True)
# (10000, 28, 28)
images = dataset.test_data.float()
labels = dataset.test_labels

features = images.view(10000, 28*28)
writer.add_embedding(features, metadata=labels,
                     label_img=images.unsqueeze(1))

writer.export_scalars_to_json('./all_scalars.json')
writer.close()
