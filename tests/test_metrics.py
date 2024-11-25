import copy
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from fedland.metrics import calculate_class_balance, path_norm


def test_class_balance(dataset: Dataset):
    loader = DataLoader(dataset=dataset)
    class_balances = calculate_class_balance(loader)

    assert class_balances.get("class_counts"), "class_counts key missing"
    assert class_balances.get("class_frequencies"), "class_frequencies key missing"
    assert class_balances.get("gini_index"), "gini_index key missing"
    gini = class_balances.get("gini_index")
    assert gini > 0.0 and gini < 1.0


def test_path_norm_finn_with_paper(dataloader: DataLoader, model: Module):
    def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
        tmp_model = copy.deepcopy(model)
        tmp_model.eval()
        with torch.no_grad():
            for param in tmp_model.parameters():
                if param.requires_grad:
                    param.abs_().pow_(p)
        data_ones = torch.ones(input_size).to(device)
        return (tmp_model(data_ones).sum() ** (1 / p)).item()

    pn_finn = path_norm(model, dataloader)
    pn_paper = lp_path_norm(model, device="cpu", input_size=[10])
    print(pn_paper)

    assert pn_finn - pn_paper < 0.001, "Mismatch in path norms"
