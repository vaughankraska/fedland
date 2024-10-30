from typing import Any, Dict, List, Self
import pymongo
import numpy as np
import torch
from pymongo.database import Database
from fedn.network.storage.statestore.stores.store import Store


def _convert_foreign_types(obj: Any) -> Any:
    """
    Convert numpy and torch objects for mongo.
    """

    # Torch objects
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().numpy().tolist()

    # Numpy objects
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_foreign_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_foreign_types(item) for item in obj]
    return obj


class ClientStat:
    def __init__(
        self,
        experiment_id: str,
        client_index: int,
        data_indices: List[int],
        balance: Dict,
        local_rounds: List[Dict] = None,
    ):
        self.experiment_id = experiment_id
        self.client_index = client_index
        self.data_indices = data_indices
        self.balance = balance
        self.local_rounds = local_rounds or []

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "client_index": self.client_index,
            "data_indices": self.data_indices,
            "balance": self.balance,
            "local_rounds": self.local_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            experiment_id=str(data["experiment_id"]),
            client_index=data["client_index"],
            data_indices=data["data_indices"],
            balance=data["balance"],
            local_rounds=data.get("local_rounds", []),
        )


class ClientStatStore(Store[ClientStat]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.database[self.collection].create_index(
            [("experiment_id", pymongo.ASCENDING), ("client_index", pymongo.ASCENDING)],
            unique=True,
        )

    def get(
        self, experiment_id: str, client_index: int, use_typing: bool = False
    ) -> ClientStat:
        """
        Get client data by experiment_id and client_index.

        Args:
            experiment_id (str): The ID of the experiment
            client_index (int): The index of the client
            use_typing (bool): Whether to return a ClientStat object or dict

        Returns:
            ClientStat or dict: The client data
        """
        response = self.database[self.collection].find_one(
            {"experiment_id": experiment_id, "client_index": client_index}
        )
        if not response:
            return None
        return ClientStat.from_dict(response) if use_typing else response

    def create_or_update(self, client_data: ClientStat) -> bool:
        """
        Creates or updates client data in the database.
        Args:
            client_data (ClientStat): The client data to create/update
        Returns:
            bool: True if operation was successful
        """
        try:
            # Convert the dictionary and handle numpy types
            client_dict = _convert_foreign_types(client_data.to_dict())

            self.database[self.collection].update_one(
                {
                    "experiment_id": client_data.experiment_id,
                    "client_index": client_data.client_index,
                },
                {"$set": client_dict},
                upsert=True,
            )
            return True
        except Exception as e:
            print(f"[!] Error updating client data: {e}")
            return False

    def append_local_round(
        self, experiment_id: str, client_index: int, local_round: Dict
    ) -> bool:
        """
        Appends a new local round to the client's training statistics.

        Args:
            experiment_id (str): The ID of the experiment
            client_index (int): The index of the client
            local_round (Dict): The training metrics for the current round

        Returns:
            bool: True if update was successful
        """
        try:
            local_round = _convert_foreign_types(local_round)
            result = self.database[self.collection].update_one(
                {"experiment_id": experiment_id, "client_index": client_index},
                {"$push": {"local_rounds": local_round}},
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"[!] Error appending local round: {e}")
            return False

    def list_by_experiment(
        self,
        experiment_id: str,
        limit: int = 1000,
        skip: int = 0,
        use_typing: bool = False,
    ) -> Dict[str, Any]:
        """
        List all client statistics for a given experiment.

        Args:
            experiment_id (str): The ID of the experiment
            limit (int): Maximum number of clients to return
            skip (int): Number of clients to skip
            use_typing (bool): Whether to return ClientStat objects or dicts

        Returns:
            Dict containing count and result list
        """
        cursor = (
            self.database[self.collection]
            .find({"experiment_id": experiment_id})
            .skip(skip)
            .limit(limit)
        )

        count = self.database[self.collection].count_documents(
            {"experiment_id": experiment_id}
        )
        result = list(cursor)

        if use_typing:
            result = [ClientStat.from_dict(item) for item in result]

        return {"count": count, "result": result}
