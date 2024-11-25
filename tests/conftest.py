import pytest
import torch
import torch.nn as nn
import torchvision
import mongomock
from typing import Generator
from torch.utils.data import Dataset, DataLoader, TensorDataset
from fedland.database_models.experiment import ExperimentStore
from pymongo.database import Database
from torchvision.datasets import VisionDataset


class MockDataset(VisionDataset):
    def __init__(self, num_samples=1000):
        super().__init__(transform=torchvision.transforms.ToTensor())
        torch.manual_seed(42)
        # 10 cols
        self.data = torch.randn(num_samples, 10)
        # 0, 1, 2 classes
        self.labels = torch.randint(0, 3, (num_samples,))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class SimpleNet(nn.Module):
    def __init__(self, in_size=10):
        super().__init__()
        self.fc1 = nn.Linear(in_size, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device("cpu")


@pytest.fixture
def model():
    return SimpleNet()


@pytest.fixture
def criterion():
    return nn.MSELoss()


@pytest.fixture
def dummy_data():
    # Create simple synthetic dataset
    X = torch.randn(10, 2)
    y = torch.randn(10, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=5)


@pytest.fixture(scope="function")
def dataset() -> Dataset:
    return MockDataset()


@pytest.fixture(scope="function")
def dataloader() -> DataLoader:
    g = torch.Generator().manual_seed(42)
    return DataLoader(dataset=MockDataset(), shuffle=False, generator=g)


@pytest.fixture(scope="session")
def mock_mongo() -> Generator[Database, None, None]:
    mock_client = mongomock.MongoClient()
    db = mock_client["mock_db"]
    yield db


@pytest.fixture(scope="function")
def experiment_store(mock_mongo: Database) -> ExperimentStore:
    return ExperimentStore(mock_mongo, "test.experiments")
