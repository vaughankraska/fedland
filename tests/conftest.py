import pytest
import torch
import mongomock
from typing import Generator
from torch.utils.data import Dataset
from fedland.database_models.experiment import ExperimentStore
from pymongo.database import Database


class MockDataset(Dataset):
    torch.manual_seed(42)

    def __init__(self, num_samples=10000):
        # 10 cols
        self.data = torch.randn(num_samples, 10)
        # 0, 1, 2 classes
        self.labels = torch.randint(0, 3, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture(scope="function")
def dataset() -> Dataset:
    return MockDataset()


@pytest.fixture(scope="session")
def mock_mongo() -> Generator[Database, None, None]:
    mock_client = mongomock.MongoClient()
    db = mock_client["mock_db"]
    yield db


@pytest.fixture(scope="function")
def experiment_store(mock_mongo: Database) -> ExperimentStore:
    return ExperimentStore(mock_mongo, "test.experiments")
