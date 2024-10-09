import os
from dotenv import load_dotenv
from fedn import APIClient
from fedn.cli.package_cmd import create_cmd

load_dotenv()
FEDN_AUTH_TOKEN = os.getenv("FEDN_AUTH_TOKEN")
FEDN_SERVER_HOST = os.getenv("FEDN_SERVER_HOST")


def setup(api: APIClient) -> dict:
    create_cmd("client/")
    # api.set_active_package(path="package.", helper="numpyhelper")
    # session = api.start_session()

    # return session


if __name__ == "__main__":
    api = APIClient(
            token=FEDN_AUTH_TOKEN,
            host=FEDN_SERVER_HOST,
            secure=True,
            verify=True
            )
    print("=>Starting Federated Run")
    print(f"[*] Clients: {api.get_clients_count()}")
    print(setup(api))
