import pytest
import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    torch.manual_seed(42)
    def __init__(self, num_samples=1000):
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
