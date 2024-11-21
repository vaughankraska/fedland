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
        self, layers=[1, 1, 1, 1], num_classes=10, channels=[64, 128, 256, 512]
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


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict):
        """InceptionBlock.
        based on: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html

        Args:
            c_in: Number of input feature maps from the previous layers
            c_red: Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out: Dictionary with keys "1x1", "3x3", "5x5", and "max"
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            nn.ReLU()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            nn.ReLU(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            nn.ReLU(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            nn.ReLU(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            nn.ReLU(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            nn.ReLU(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

class CifarInception(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self._create_network()
        self._init_params()

    def _create_network(self):
        # Reduced initial channels from 64 to 32
        self.input_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.inception_blocks = nn.Sequential(
            # First block group - 32x32
            InceptionBlock(
                32,
                c_red={"3x3": 16, "5x5": 8},
                c_out={"1x1": 8, "3x3": 16, "5x5": 4, "max": 4},
            ),
            InceptionBlock(
                32,
                c_red={"3x3": 16, "5x5": 8},
                c_out={"1x1": 12, "3x3": 24, "5x5": 6, "max": 6},
            ),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Second block group - 16x16
            InceptionBlock(
                48,
                c_red={"3x3": 16, "5x5": 8},
                c_out={"1x1": 12, "3x3": 24, "5x5": 6, "max": 6},
            ),
            InceptionBlock(
                48,
                c_red={"3x3": 16, "5x5": 8},
                c_out={"1x1": 16, "3x3": 24, "5x5": 8, "max": 8},
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8

            # Final block group - 8x8
            InceptionBlock(
                56,  # Total channels from previous block (16+24+8+8=56)
                c_red={"3x3": 24, "5x5": 8},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
            ),
        )

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.num_classes)  # 64 = 16+32+8+8 from last inception block
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
