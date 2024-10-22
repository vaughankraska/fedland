import pytest
from torch.utils.data import Dataset, DataLoader
from fedland.metrics import pac_bayes_bound, path_norm, frobenius_norm, calculate_class_balance

def test_class_balance(dataset: Dataset):
    loader = DataLoader(dataset=dataset)
    class_balances = calculate_class_balance(loader)

    assert class_balances.get('class_counts'), 'class_counts key missing'
    assert class_balances.get('class_frequencies'), (
            'class_frequencies key missing')
    assert class_balances.get('gini_index'), 'gini_index key missing'
    gini = class_balances.get("gini_index")
    assert gini > 0.0 and gini < 1.0


@pytest.mark.xfail(reason="Unimplemented")
def test_path_norm(dataset: Dataset):
    assert False, "TODO"


@pytest.mark.xfail(reason="Unimplemented")
def test_pac_bayes(dataset: Dataset):
    assert False, "TODO"


@pytest.mark.xfail(reason="Unimplemented")
def test_frobenius(dataset: Dataset):
    assert False, "TODO"
