# based on Horvath Csongor's MSC code, adapted for MNIST and a more simple NN
import torch
import torch.nn as nn
import torch.optim as optim
from fed_land.loaders import load_mnist_data
from fed_land.networks import FedNet
from fed_land.metrics import path_norm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    epochs = 10
    learning_rate = 0.1
    momentum = 0.5

    train_loader, test_loader = load_mnist_data(batch_size)
    # Counterintuitive that we are running FedNet in centralized setting
    # but welp, I like the name.
    model: nn.Module = FedNet().to(DEVICE)
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
