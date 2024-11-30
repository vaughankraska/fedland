import torch
import json
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from torch.utils.data import DataLoader
from fedland.loaders import PartitionedDataLoader
import torchvision
from fedland.networks import CifarFedNet
from fedland.metrics import evaluate, path_norm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> Dict[str, List[float]]:
    model.to(device)
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "path_norm": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 200 == 199:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "  # noqa E501
                    f"Loss: {running_loss/100:.3f}, Accuracy: {100.*correct/total:.2f}%"
                )  # noqa E501
                running_loss, correct, total = 0.0, 0, 0

        # Train eval
        train_loss, train_acc = evaluate(model, train_loader, criterion, device)  # noqa E501
        results["train_loss"].append(float(train_loss))
        results["train_accuracy"].append(float(train_acc))

        # Test eval
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results["test_loss"].append(float(test_loss))
        results["test_accuracy"].append(float(test_acc))

        # And path norm
        pn = path_norm(model, train_loader)
        results["path_norm"].append(float(pn))

        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:.2f}%")  # noqa E501
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%")
        print(f"Path Norm: {pn:.2f}")
        print("=" * 50)

    return results


def dump(data: dict, base_filename: str, model: nn.Module):
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"results/{date_str}_{base_filename}.json"
    data["model"] = model.__repr__()
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Data saved to {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=>Starting Centralized Run on {device}")
    # Note! Mismatch?
    # From Horvaths code:
    # BATCH_SIZE = 128 ...
    # optimizer_name = 'SGD'
    # optimizer_params = {'lr' : 0.02, 'momentum' : 0.9, 'weight_decay': 0}
    # grad_clip = 0.01
    # But paper says: "For ResNet with CIFAR10 to obtain good minima β = 0.5, b = 0.1 were used for 10 epochs."

    batch_size = 32
    epochs = 20
    learning_rate = 0.01
    momentum = 0.5

    # train_loader, test_loader = load_mnist_data()

    OUT_DIR = "./temp"
    train_set = torchvision.datasets.CIFAR10(
        root=f"{OUT_DIR}/train",
        transform=torchvision.transforms.ToTensor(),
        train=True,
        download=True,
    )
    train_loader = PartitionedDataLoader(train_set, num_partitions=1, partition_index=0)
    test_set = torchvision.datasets.CIFAR10(
        root=f"{OUT_DIR}/test",
        transform=torchvision.transforms.ToTensor(),
        train=False,
        download=True,
    )
    test_loader = PartitionedDataLoader(test_set, num_partitions=1, partition_index=0)

    model: nn.Module = CifarFedNet(num_classes=10)
    # model: nn.Module = torchvision.models.resnet18(weights=None, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    results = train(
        model, train_loader, test_loader, criterion, optimizer, device, epochs
    )
    dump(results, "centalized_cifar_10_fednet", model)
    print("<== Training Finished")
