# based on Horvath Csongor's MSC code, adapted for MNIST and a more simple NN
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Matches fedn's pytorch example
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def load_data(batch_size=128, shuffle=True) -> tuple[DataLoader, DataLoader]:
    # Make dir if necessary
    out_dir = "./data/centralized"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    train_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/train",
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True,
            )
    test_set = torchvision.datasets.MNIST(
            root=f"{out_dir}/test",
            transform=torchvision.transforms.ToTensor(),
            train=False,
            download=True,
            )

    train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle
            )
    test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle
            )

    return train_loader, test_loader


def train(
        model: nn.Module,
        train_loader,
        criterion,
        optimizer: optim.Optimizer,
        epochs=5
        ):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0


def path_norm(model: nn.Module, in_size: torch.Tensor):
    print(f"in_size={in_size}")
    modified_model = copy.deepcopy(model)
    modified_model.to(torch.device("cpu"))

    with torch.no_grad():
        for param in modified_model.parameters():
            param.data = param.data ** 2
            print(f"param data: {param.data}")

    ones = torch.ones_like(in_size)
    print(f"model: {modified_model}")

    t = modified_model.forward(ones)
    print(f"forwarded: {t}")
    temp = (torch.sum(t).data)
    print(f"temp: {temp}")
    return temp ** 0.5


if __name__ == "__main__":

    print(f"=>Starting Centralized Run on {DEVICE}")
    # Note! Mismatch?
    # From Horvaths code:
    # BATCH_SIZE = 128 ...
    # optimizer_name = 'SGD'
    # optimizer_params = {'lr' : 0.02, 'momentum' : 0.9, 'weight_decay': 0}
    # grad_clip = 0.01
    # But paper says: "For ResNet with CIFAR10 to obtain good minima β = 0.5, b = 0.1 were used for 10 epochs."

    batch_size = 64
    epochs = 5 # 10!
    learning_rate = 0.1
    momentum = 0.5

    train_loader, test_loader = load_data(batch_size)
    model: nn.Module = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum
            )

    train(model, train_loader, criterion, optimizer, epochs)
    print("<== Training Finished")
    # in_size = (batch_size, 784)
    input, out = next(iter(train_loader))
    p_norm = path_norm(model, input[0].unsqueeze(0))
    print(f"\nPath Norm: {p_norm}")
