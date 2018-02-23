import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Data():
    def __init__(self, args):
        data_dir = 'dataset/fashionmnist'
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                                   transform=train_transform,
                                                   download=True)
        self.test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                                  transform=test_transform)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False)

    def embed(self, writer):
        # (10000, 28, 28)
        images = self.test_dataset.test_data.float()
        labels = self.test_dataset.test_labels

        features = images.view(10000, 28*28)
        writer.add_embedding(features, metadata=labels,
                             label_img=images.unsqueeze(1))

    def load(self):
        return self.train_loader, self.test_loader
