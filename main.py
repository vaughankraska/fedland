# This file is an entry point to run all the experiments
import os
from typing import List
import time
import sys
import tarfile
import json
import subprocess
import signal
import numpy as np
from datetime import datetime
import uuid
from fedn import APIClient
from fedn.cli.run_cmd import check_yaml_exists
from fedn.utils.dispatcher import _read_yaml_file, Dispatcher
from fedland.loaders import DatasetIdentifier
from fedland.database_models.experiment import Experiment
from fedland.networks import ModelIdentifier


# CONSTANTS
ROUNDS = 30
CLIENT_LEVEL = 2
EXPERIMENTS = [
    Experiment(
        id=str(uuid.uuid4()),
        description="EXAMPLE: ResNet CIFAR-10, even classes",
        dataset_name=DatasetIdentifier.CIFAR.value,
        model=ModelIdentifier.CIFAR_RESNET.value,
        timestamp=datetime.now().isoformat(),
        target_balance_ratios=[
            [0.01]*10,
            #[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.22, 0.20, 0.10], # [0.01]*10
            [
                float(x)
                for x in (
                    np.exp(-0.07 * np.arange(10)) / sum(np.exp(-0.07 * np.arange(10)))
                )
            ],
            
        ],
        subset_fractions=[1.0, 1.0], # control the amount of data each client gets
        client_stats=[],
        aggregator="fedopt",  # OR "fedopt" / "fedavg"
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


def create_experiment(experiment: Experiment):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = current_dir + "/results/experiments.json"
    os.environ.setdefault("RESULTS_DIR", current_dir + "/results")

    if os.path.exists(filename):
        with open(filename, "r+") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(experiment.to_dict())
            else:
                existing_data = [existing_data, experiment.to_dict()]
            f.seek(0)
            json.dump(existing_data, f, indent=2)
    else:
        with open(filename, "w") as f:
            json.dump([experiment.to_dict()], f, indent=2)


def setup(api: APIClient, experiment: Experiment) -> dict:
    os.environ.setdefault("TEST_ID", experiment.id)
    create_experiment(experiment)
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


class ClientManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start_client(self, client_number: str, experiment_id: str):
        env = os.environ.copy()
        env.update(
            {
                "RESULTS_DIR": f"{os.path.dirname(os.path.abspath(__file__))}/results",
                "TEST_ID": f"{experiment_id}",
                "CLIENT_ID": f"{client_number}",
            }
        )
        process = subprocess.Popen(
            [
                "fedn",
                "client",
                "start",
                "-n",
                client_number,
                "--init",
                "settings-client-local.yaml",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.processes.append(process)

    def start_multiple_clients(self, num_clients: int, experiment_id: str):
        for i in range(num_clients):
            self.start_client(str(i), experiment_id)
            print(f"[*], Client {i} started")
            time.sleep(7)  # Wait for client to start 1

    def signal_handler(self, signum, frame):
        print("[*] Shutting down clients")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()

    def peak_processes(self):
        try:
            running = sum(1 for p in self.processes if p.poll() is None)
            print(f"[*] Running {running} clients")
        except KeyboardInterrupt:
            self.cleanup()


if __name__ == "__main__":
    api = APIClient(
        "localhost",
        8092,
        secure=False,
    )
    # manager = ClientManager()

    for experiment in EXPERIMENTS:
        experiment.timestamp = datetime.now().isoformat()
        # Set package and seed and test in experiments.json
        setup(api, experiment)

        date_str = datetime.now().strftime("%Y%m%d%H%M")
        sesh_id = f"sesh-{date_str}"
        sesh_config = {
            "id": sesh_id,
            "min_clients": CLIENT_LEVEL,
            "rounds": ROUNDS,
            "round_timeout": 600,
            "aggregator": experiment.aggregator,
            "aggregator_kwargs": experiment.aggregator_kwargs,
        }
        session = api.start_session(**sesh_config) 
        #time.sleep(10) # Wait for session to start we can #
        #manager.start_multiple_clients(CLIENT_LEVEL, experiment.id)
        while not session["success"]:
            print(f"=X Waiting to start run ({session['message']})")
            # manager.peak_processes()
            time.sleep(4)
            session = api.start_session(**sesh_config)
        print(f"=>Started Federated Session:\n{session}")
        print(f"=>Experiment:\n{experiment.description}\nid:{experiment.id}")
        while not api.session_is_finished(sesh_id):
            status = api.get_session_status(sesh_id)
            print(f"[*] Status: {status}")
            time.sleep(60)
        # manager.cleanup()
