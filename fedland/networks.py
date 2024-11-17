import torch
import torch.nn as nn
import torch.nn.functional as fun
from torchvision.models.resnet import ResNet, BasicBlock


class FedNet(nn.Module):
    """
    For MNIST Dataset,
    A fully connected NN to match FEDn's pytorch example.
    Note the output is not softmaxed.
    """

    def __init__(self, num_pixels=784):
        super(FedNet, self).__init__()
        self.num_pixels = num_pixels
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x) -> torch.Tensor:
        x = fun.relu(self.fc1(x.reshape(x.size(0), self.num_pixels)))
        x = fun.dropout(x, p=0.5, training=self.training)
        x = fun.relu(self.fc2(x))
        x = fun.relu(self.fc3(x))
        # x = fun.log_softmax(self.fc3(x), dim=1)

        return x


class CifarFedNet(nn.Module):
    """For CIFAR Dataset"""

    def __init__(self, num_classes=10):
        super(CifarFedNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x) -> torch.Tensor:
        x = self.pool(fun.relu(self.bn1(self.conv1(x))))
        x = self.pool(fun.relu(self.bn2(self.conv2(x))))
        x = self.pool(fun.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 64 * 4 * 4)

        x = self.dropout(fun.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class CifarResNet(ResNet):
    def __init__(
        self, layers=[1, 1, 1, 1], num_classes=100, channels=[64, 128, 256, 512]
    ):
        super().__init__(block=BasicBlock, layers=layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        if channels != [64, 128, 256, 512]:
            self.inplanes = channels[0]
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0])
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
            self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], stride=2)
            self.layer4 = self._make_layer(BasicBlock, channels[3], layers[3], stride=2)
            self.fc = nn.Linear(channels[3] * BasicBlock.expansion, num_classes)
