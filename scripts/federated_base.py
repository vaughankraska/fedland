import os
import copy
import tarfile
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from fedn import APIClient
from fedn.network.clients.client import Client

load_dotenv()
FEDN_AUTH_TOKEN = os.getenv("FEDN_AUTH_TOKEN")
FEDN_AUTH_REFRESH_TOKEN = os.getenv("FEDN_AUTH_REFRESH_TOKEN")
FEDN_SERVER_HOST = os.getenv("FEDN_SERVER_HOST")
FEDN_NUM_DATA_SPLITS = int(os.getenv("FEDN_NUM_DATA_SPLITS", 5))

client_config = {
    "client_id": None,
    "name": "client",
    "discover_host": FEDN_SERVER_HOST,
    "discover_port": None,
    "token": FEDN_AUTH_TOKEN,
    "refresh_token": FEDN_AUTH_REFRESH_TOKEN,
    "force_ssl": True,
    "dry_run": False,
    "secure": True,
    "heartbeat_interval": 2,
    "reconnect_after_missed_heartbeat": 30,
    "combiner": None,
    "remote_compute_context": None,
    "verify": None,
    "preferred_combiner": None,
}


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
    assert os.path.exists("seed.npz"), "seed.npz not found, have you run `fedn run build --path client`?"
    create_cmd()  # package.tgz

    res = api.set_active_package(path="package.tgz", helper="numpyhelper")
    print(f"[*] {res.get('message')}")
    res = api.set_active_model("seed.npz")
    print(f"[*] {res.get('message')}")


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y%m%d%M")
    api = APIClient(
            token=FEDN_AUTH_TOKEN,
            host=FEDN_SERVER_HOST,
            secure=True,
            verify=True
            )
    print("=>Starting Federated Run")
    # print(f"[*] Clients: {api.get_clients_count()}")
    # setup(api)

    # downloads_path = os.path.expanduser("~/Downloads") # Read from downloads clientx.yaml?
    # clients = []
    # for i in range(FEDN_NUM_DATA_SPLITS):
    #     config = copy.deepcopy(client_config)
    #     config["client_id"] = f"client-{i}-{date_str}"
    #     config["name"] = f"client-{i}"
    #     # config["combiner"] = combiner
    #     client = Client(config)
    #     clients.append(client)

    # print(date_str)
    # print(f"clients: {clients}")
    # time.sleep(120)
    api.start_session(f"sesh-{date_str}")
