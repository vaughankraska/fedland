# This file is an entry point to run all the experiments
import os
import numpy as np
from typing import List
import time
import sys
import tarfile
import json
import subprocess
import threading
import signal
from datetime import datetime
import uuid
from fedn import APIClient
from fedn.cli.run_cmd import check_yaml_exists
from fedn.utils.dispatcher import _read_yaml_file, Dispatcher
from fedland.loaders import DatasetIdentifier
from fedland.database_models.experiment import Experiment
from fedland.networks import ModelIdentifier


# CONSTANTS
ROUNDS = 60
CLIENT_LEVEL = 5
SUBSET_FRACTIONS = [1, 1, 0.7, 0.5, 0.05]
CLASS_IMBALANCE = [
        [0.1] * 10,
        [0.1] * 10,
        [float(x) for x in (np.exp(-0.5 * np.arange(10)) / sum(np.exp(-0.5 * np.arange(10))))],
        [float(x) for x in reversed(np.exp(-0.5 * np.arange(10)) / sum(np.exp(-0.5 * np.arange(10))))],
        [float(x) for x in (np.exp(-0.7 * np.arange(10)) / sum(np.exp(-0.7 * np.arange(10))))]
        ]
EXPERIMENTS = [
    Experiment(
        id=str(uuid.uuid4()),
        description="FedNet CIFAR-10, 5 clients, IID, balanced, fedavg",
        dataset_name=DatasetIdentifier.CIFAR.value,
        model=ModelIdentifier.CIFAR_FEDNET.value,
        timestamp=datetime.now().isoformat(),
        client_stats=[],
        aggregator="fedavg",
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
        self.process_outputs: List[str] = []
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
            text=True,
            env=env,
        )
        self.processes.append(process)
        threading.Thread(
                target=self.read_output,
                args=(process, client_number),
                daemon=True
                ).start()

    def read_output(self, process: subprocess.Popen, client_number: str):
        try:
            for line in process.stdout:
                print(f"C{client_number}:[*] {line.strip()}")
            for line in process.stderr:
                print(f"C{client_number}:[!] {line.strip()}")
        except Exception as e:
            print(f"Error reading output for Client {client_number}: \n {e}")
        finally:
            process.wait()

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
    manager = ClientManager()

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
        time.sleep(5)

        manager.start_multiple_clients(CLIENT_LEVEL, experiment.id)
        while not session["success"]:
            print(f"=X Waiting to start run ({session['message']})")
            manager.peak_processes()
            time.sleep(4)
            session = api.start_session(**sesh_config)
        print(f"=>Started Federated Session:\n{session}")
        print(f"=>Experiment:\n{experiment.description}\nid:{experiment.id}")
        while not api.session_is_finished(sesh_id):
            status = api.get_session_status(sesh_id)
            print(f"[*] Status: {status}")
            time.sleep(60)
        time.sleep(30)
        manager.cleanup()
