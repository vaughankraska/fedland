# This file is an entry point to run all the experiments in /scripts
import os
import tarfile
from datetime import datetime
from fedn import APIClient
from fedland.database_models.experiment import ExperimentStore, Experiment, experiment_store
from fedland.database_models.client_stat import ClientStat
from pymongo.database import Database


session_config = {
        "helper": "numpyhelper",
        "id": datetime.now().strftime("%Y%m%d%M"),
        "aggregator": "fedopt",
        "rounds": 100,
        "validate": False,
        }
# TODO: init, experiments from here, add clients/rounds from each client
EXPERIMENTS = [
        {
            "description": "Experiment 1",
            "active_clients": 0,
            "learning_rate": 0.01,
            "dataset_name": "MNIST",
            "model": "CNN",
            "clients": [
                {
                    "data_indices": [1, 2, 8, 11],
                    "balance": {
                        "class_counts": {0: 200, 1: 150, 2: 100, 3: 50},
                        "class_frequencies": {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
                        "gini_index": 0.6,
                        },
                    "local_rounds": [
                        {
                            "session_id": "session_1",
                            "epoch": 1,
                            "loss": 0.8,
                            "path_norm": 0.1,
                            "pac_bayes_bound": 0.05,
                            "frobenius_norm": 0.2,
                            },
                        {
                            "session_id": "session_1",
                            "epoch": 2,
                            "loss": 0.6,
                            "path_norm": 0.08,
                            "pac_bayes_bound": 0.03,
                            "frobenius_norm": 0.15,
                            },
                        ],
                    }
                ],
            }
        ]


def create_example_experiment(database: Database):
    # database: "from fedn.network.api.v1.shared import mdb"
    # Create client statistics
    client_stat = ClientStat(
        experiment_id="",  # Will be set after experiment creation
        client_index=0,
        data_indices=[1, 2, 8, 11],
        balance={
            "class_counts": {0: 200, 1: 150, 2: 100, 3: 50},
            "class_frequencies": {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1},
            "gini_index": 0.6,
        },
        local_rounds=[]
    )

    # Create experiment with client statistics
    experiment = Experiment(
        id="",  # Will be set by MongoDB
        description="Experiment 1",
        dataset_name="MNIST",
        model="FedNet",
        timestamp=datetime.now().isoformat(),
        active_clients=1,
        learning_rate=0.1,
        client_stats=[]  # insert client stats as pleased
    )

    # Store the experiment and its client statistics
    experiment_id = experiment_store.create_experiment(experiment)

    # Append a new local round
    new_local_round = {
        "session_id": "session_1",
        "epoch": 2,
        "loss": 0.6,
        "path_norm": 0.08,
        "pac_bayes_bound": 0.03,
        "frobenius_norm": 0.15,
    }

    experiment_store.client_stat_store.append_local_round(
        experiment_id=experiment_id,
        client_index=0,
        local_round=new_local_round
    )

    return experiment_id


def create_cmd(name="package.tgz") -> str:
    # Copied from FEDn cli create_cmd() since there is no
    # export/import of the logic without the CLI
    path = os.path.abspath("client/")
    yaml_file = os.path.join(path, "fedn.yaml")
    if not os.path.exists(yaml_file):
        print(f"[!] Could not find fedn.yaml in {path}")
        exit(-1)

    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
        print(f"[*] Created package {name}")

    return name


def setup(api: APIClient) -> dict:
    # ensure seed.npz from `fedn run build --path client`
    assert os.path.exists(
        "package.tgz"
    ), "package.tgz not found, have you run create_cmd OR `fedn package create --path client`?"
    assert os.path.exists(
        "seed.npz"
    ), "seed.npz not found, have you run `fedn run build --path client`?"
    create_cmd()  # package.tgz (Recompile package)

    res = api.set_active_package(path="package.tgz", helper="numpyhelper")
    print(f"[*] {res.get('message')}")
    res = api.set_active_model("seed.npz")
    print(f"[*] {res.get('message')}")


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y%m%d%M")
    api = APIClient(
        "localhost",
        8092,
        secure=False,
    )
    # Set package and seed
    # setup(api)
    # session = api.start_session(f"sesh-{date_str}", min_clients=2)
    # print(f"=>Started Federated Run:\n{session}")
