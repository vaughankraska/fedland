# based on Horvath Csongor's MSC code, adapted for MNIST and a more simple NN
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision


# Matches fedn's pytorch example
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = fun.relu(self.fc1(x.reshape(x.size(0), 784)))
        x = fun.dropout(x, p=0.5, training=self.training)
        x = fun.relu(self.fc2(x))
        x = fun.log_softmax(self.fc3(x), dim=1)

        return x


def load_data(batch_size=128, shuffle=True):
    # Make dir if necessary
    out_dir = "./data/centralized"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    train_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/train",
            transform=torchvision.transforms.ToTensor,
            train=True,
            download=True
            )
    test_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/test",
            transform=torchvision.transforms.ToTensor,
            train=False,
            download=True
            )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def path_norm(model, in_size):
    todo!()
