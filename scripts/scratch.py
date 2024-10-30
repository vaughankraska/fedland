from datetime import datetime
from fedland.database_models.experiment import experiment_store, Experiment
from fedland.loaders import DatasetIdentifier


def create_test_experiment():
    experiment = Experiment(
        id="",
        description="Testy Experiment 1 CIFAR",
        dataset_name="",
        model=DatasetIdentifier.MNIST.value,
        timestamp=datetime.now().isoformat(),
        active_clients=3,
        learning_rate=0.1,
        client_stats=[],
    )

    # Store the experiment and its client statistics
    return experiment_store.create_experiment(experiment)


if __name__ == "__main__":
    print("=>Running Scratch")
    id = create_test_experiment()
    print(f"test experiment id={id}")
