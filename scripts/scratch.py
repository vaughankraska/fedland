from datetime import datetime
from fedland.database_models.experiment import experiment_store, Experiment
from fedland.loaders import DatasetIdentifier


def create_test_experiment():
    experiment = Experiment(
        id="",
        description="Testy Experiment updated",
        dataset_name=DatasetIdentifier.MNIST.value,
        model="FedNet",
        timestamp=datetime.now().isoformat(),
        learning_rate=0.1,
        target_balance_ratios=[
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        subset_fractions=[1, 1, 1],
        client_stats=[],
    )

    # Store the experiment and its client statistics
    return experiment_store.create_experiment(experiment)


if __name__ == "__main__":
    print("=>Running Scratch")
    id = create_test_experiment()
    print(f"test experiment id={id}")
