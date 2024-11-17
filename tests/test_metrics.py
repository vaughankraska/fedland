from torch.utils.data import Dataset, DataLoader
from fedland.metrics import calculate_class_balance


def test_class_balance(dataset: Dataset):
    loader = DataLoader(dataset=dataset)
    class_balances = calculate_class_balance(loader)

    assert class_balances.get("class_counts"), "class_counts key missing"
    assert class_balances.get("class_frequencies"), "class_frequencies key missing"
    assert class_balances.get("gini_index"), "gini_index key missing"
    gini = class_balances.get("gini_index")
    assert gini > 0.0 and gini < 1.0
