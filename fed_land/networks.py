import torch
import torch.nn as nn
import torch.nn.functional as fun


class FedNet(nn.Module):
    """
    A fully connected NN to match FEDn's pytorch example.
    Note the output is not softmaxed.
    """

    def __init__(self):
        super(FedNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x) -> torch.Tensor:
        x = fun.relu(self.fc1(x.reshape(x.size(0), 784)))
        x = fun.dropout(x, p=0.5, training=self.training)
        x = fun.relu(self.fc2(x))
        x = fun.relu(self.fc3(x))
        # x = fun.log_softmax(self.fc3(x), dim=1)

        return x
