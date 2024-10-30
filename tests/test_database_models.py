import pytest
from datetime import datetime
from fedland.database_models.experiment import Experiment, ExperimentStore
from fedland.database_models.client_stat import ClientStat


def test_create_empty_experiment(experiment_store: ExperimentStore):
    with pytest.raises(TypeError):
        Experiment()


def test_create_empty_experiment_from_dict(experiment_store: ExperimentStore):
    experiment = {}
    with pytest.raises(KeyError):
        experiment = Experiment.from_dict(experiment)


def test_create_experiment(experiment_store: ExperimentStore) -> None:
    client_stat = ClientStat(
        experiment_id="",
        client_index=0,
        data_indices=[1, 2, 8, 11],
        balance={
            "class_counts": {"class_0": 200, "class_1": 150},
            "class_frequencies": {"class_0": 0.4, "class_1": 0.3},
            "gini_index": 0.6,
        },
        local_rounds=[]
    )

    # Create experiment with client statistics
    experiment = Experiment(
        id="",  # Will be set by MongoDB?
        description="Test Experiment",
        dataset_name="MNIST",
        model="FedNet",
        timestamp=datetime.now().isoformat(),
        active_clients=1,
        learning_rate=0.1,
        client_stats=[client_stat]  # insert client stats as pleased
    )

    experiment_id: str = experiment_store.create_experiment(experiment)
    latest_experiment = experiment_store.get_latest()
    created_experiment = experiment_store.get(experiment_id)

    assert created_experiment is not None
    assert latest_experiment is not None
    assert latest_experiment.id == experiment_id
    assert created_experiment["description"] == "Test Experiment"
    assert created_experiment["client_stats"][0] is not None
