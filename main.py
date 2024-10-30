# This file is an entry point to run all the experiments in /scripts
import os
import time
import tarfile
from datetime import datetime
from fedn import APIClient


session_config = {
        "helper": "numpyhelper",
        "id": "todo",  # "sesh-" + datetime.now().strftime("%Y%m%d%M"),
        "aggregator": "fedavg",
        "rounds": 100,
        "validate": False,
        "min_clients": "todo",  # clients get scaled in docker (manual)
        }
# TODO: init, experiments from here, add clients/rounds from each client (inspo GridSearchCV)
EXPERIMENTS = [
        {
            "description": "Experiment 1",
            "dataset_name": "MNIST",
            "model": "CNN",
            "timestamp": datetime.now(),
            "active_clients": 0,
            "learning_rate": 0.01,
            "target_balance_ratios": None,
            "subset_fractions": None,
            "client_stats": None,
            }
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
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    sesh_id = f"sesh-{date_str}"
    api = APIClient(
        "localhost",
        8092,
        secure=False,
    )
    # Set package and seed
    setup(api)
    session = api.start_session(sesh_id, min_clients=2)
    while not session["success"]:
        print(f"=X Waiting to start run ({session['message']})")
        time.sleep(4)
        session = api.start_session(sesh_id, min_clients=2)
    print(f"=>Started Federated Run:\n{session}")
