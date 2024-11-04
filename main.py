# This file is an entry point to run all the experiments
import os
import time
import tarfile
from datetime import datetime
from fedn import APIClient
from fedland.loaders import DatasetIdentifier
from fedland.database_models.experiment import experiment_store, Experiment


# CONSTANTS
ROUNDS = 30
CLIENT_LEVEL = 2
LEARNING_RATE = 0.1  # (default in Experiment)
EXPERIMENTS = [
        Experiment(
            id="",
            description='TESTING NON SEEDS, uneven classes',
            dataset_name=DatasetIdentifier.MNIST.value,
            model="FedNet",
            timestamp=datetime.now().isoformat(),
            target_balance_ratios=[
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.5, 0.25, 0.125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ],
            client_stats=[],
            ),
        # Experiment(
        #     id="",
        #     description='Three Clients, one client even classes, second client not (pareto-like classes), third with only two classes.',
        #     dataset_name=DatasetIdentifier.MNIST.value,
        #     model="FedNet",
        #     timestamp=datetime.now().isoformat(),
        #     target_balance_ratios=[
        #         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        #         [0.5, 0.25, 0.125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         ],
        #     client_stats=[],
        #     ),
        # Experiment(
        #     id="",
        #     description="Three Clients, offsetting classes ie 1-3, the second has 3-6, third has rest.",
        #     dataset_name=DatasetIdentifier.MNIST.value,
        #     model="FedNet",
        #     timestamp=datetime.now().isoformat(),
        #     target_balance_ratios=[
        #         [0.33, 0.33, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         [0.00, 0.00, 0.00, 0.33, 0.33, 0.34, 0.0, 0.0, 0.0, 0.0],
        #         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.25, 0.25, 0.25],
        #         ],
        #     client_stats=[],
        #     ),
        ]


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
    api = APIClient(
        "localhost",
        8092,
        secure=False,
    )
    # Set package and seed
    # setup(api)
    # time.sleep(5)

    # def get_user_confirmation():
    #     while True:
    #         response = input("\n[y]/n Continue with next experiment? ").lower().strip()
    #         if response in ['', 'y']:
    #             return True
    #         elif response == 'n':
    #             return False
    #         print("Please enter 'y' or 'n' (or press Enter for yes)")

    for experiment in EXPERIMENTS:
        # Set package and seed
        setup(api)
        time.sleep(5)

        experiment.timestamp = datetime.now().isoformat()
        exp_id = experiment_store.create_experiment(experiment)
        assert exp_id is not None, "Cannot start Experiment without Experiment"

        time.sleep(5)

        # if not get_user_confirmation():
        #     print("Stopping experiments.")
        #     break

        date_str = datetime.now().strftime("%Y%m%d%H%M")
        sesh_id = f"sesh-{date_str}"
        sesh_config = {
                "id": sesh_id,
                "min_clients": CLIENT_LEVEL,
                "rounds": ROUNDS,
                }
        session = api.start_session(**sesh_config)
        while not session["success"]:
            print(f"=X Waiting to start run ({session['message']})")
            time.sleep(4)
            session = api.start_session(**sesh_config)
        print(f"=>Started Federated Session:\n{session}")
        print(f"=>Experiment:\n{experiment.description}\nid:{exp_id}")
        while not api.session_is_finished(sesh_id):
            status = api.get_session_status(sesh_id)
            print(f"[*] Status: {status}")
            time.sleep(35)
