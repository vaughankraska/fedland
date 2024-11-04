# This file is an entry point to run all the experiments
import os
import time
import tarfile
from datetime import datetime
from fedn import APIClient
from fedn.cli.run_cmd import check_yaml_exists
from fedn.utils.dispatcher import _read_yaml_file, Dispatcher
from fedland.loaders import DatasetIdentifier
from fedland.database_models.experiment import experiment_store, Experiment


# CONSTANTS
ROUNDS = 10
CLIENT_LEVEL = 2
LEARNING_RATE = 0.1  # (default in Experiment)
EXPERIMENTS = [
        Experiment(
            id="",
            description='TESTING CIFAR, even classes',
            dataset_name=DatasetIdentifier.CIFAR.value,
            model="CifarFedNet",
            timestamp=datetime.now().isoformat(),
            target_balance_ratios=[
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ],
            client_stats=[],
            ),
        ]


def create_cmd(name="package.tgz") -> str:
    """Copied from FEDn cli (same as `fedn package create --path client`)"""
    path = os.path.abspath("client/")
    yaml_file = os.path.join(path, "fedn.yaml")
    if not os.path.exists(yaml_file):
        print(f"[!] Could not find fedn.yaml in {path}")
        exit(-1)

    with tarfile.open(name, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
        print(f"[*] Created package {name}")

    return name


def build_cmd(path="client/"):
    """Copied from FEDn cli (same as `fedn package create --path client`)"""
    path = os.path.abspath(path)
    yaml_file = check_yaml_exists(path)

    config = _read_yaml_file(yaml_file)
    # Check that build is defined in fedn.yaml under entry_points
    if "build" not in config["entry_points"]:
        raise ValueError("No build command defined in fedn.yaml")

    dispatcher = Dispatcher(config, path)
    _ = dispatcher._get_or_create_python_env()
    dispatcher.run_cmd("build")


def setup(api: APIClient) -> dict:

    create_cmd()  # package.tgz (Recompile package)
    build_cmd()  # seed.npz (Reinit seed model)
    assert os.path.exists(
        "package.tgz"
    ), "package.tgz not found, have you run create_cmd OR `fedn package create --path client`?"
    assert os.path.exists(
        "seed.npz"
    ), "seed.npz not found, have you run build_cmd `fedn run build --path client`?"

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

    # def get_user_confirmation():
    #     while True:
    #         response = input("\n[y]/n Continue with next experiment? ").lower().strip()
    #         if response in ['', 'y']:
    #             return True
    #         elif response == 'n':
    #             return False
    #         print("Please enter 'y' or 'n' (or press Enter for yes)")

    for experiment in EXPERIMENTS:

        experiment.timestamp = datetime.now().isoformat()
        exp_id = experiment_store.create_experiment(experiment)
        assert exp_id is not None, "Cannot start Experiment without Experiment"

        # Set package and seed
        setup(api)
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
