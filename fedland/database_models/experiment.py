from typing import Dict, List, Self, Optional
from typing_extensions import override
import pymongo
from pymongo.database import Database
from fedland.database_models.client_stat import ClientStat, ClientStatStore
from fedn.network.storage.statestore.stores.store import Store
from fedn.network.api.v1.shared import mdb


class Experiment:
    def __init__(
        self,
        id: str,
        description: str,
        dataset_name: str,
        model: str,
        timestamp: str,
        active_clients: int = 0,
        learning_rate: float = 0.1,
        target_balance_ratios: List[List[float]] = None,
        subset_fractions: List[float] = None,
        client_stats: List[ClientStat] = None,
    ):
        self.id = id
        self.description = description
        self.dataset_name = dataset_name
        self.model = model
        self.timestamp = timestamp
        self.active_clients = active_clients
        self.learning_rate = learning_rate
        self.target_balance_ratios = target_balance_ratios
        self.subset_fractions = subset_fractions
        self.client_stats = client_stats or []

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "dataset_name": self.dataset_name,
            "model": self.model,
            "timestamp": self.timestamp,
            "active_clients": self.active_clients,
            "learning_rate": self.learning_rate,
            "target_balance_ratios": self.target_balance_ratios,
            "subset_fractions": self.subset_fractions,
            "client_stats": [stat.to_dict() for stat in self.client_stats],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        client_stats = []
        if "client_stats" in data:
            client_stats = [
                    ClientStat.from_dict(stat) for stat in data["client_stats"]
                    ]

        return cls(
            id=str(data["_id"]),
            description=data.get("description"),
            dataset_name=data.get("dataset_name"),
            model=data.get("model"),
            timestamp=data.get("timestamp"),
            active_clients=data.get("active_clients", 0),
            learning_rate=data.get("learning_rate", 0.1),
            target_balance_ratios=data.get("target_balance_ratios", None),
            subset_fractions=data.get("subset_fractions", None),
            client_stats=client_stats
        )


class ExperimentStore(Store[Experiment]):
    def __init__(self, database: Database, collection: str):
        super().__init__(database, collection)
        self.client_stat_store = ClientStatStore(database, "local.client_stats")

    def create_experiment(self, experiment: Experiment) -> str:
        """
        Creates a new experiment in the database.
        Args:

            experiment (Experiment): The experiment object to create

        Returns:
            str: The ID of the created experiment
        """
        experiment_dict = experiment.to_dict()
        result = self.collection.insert_one(experiment_dict)

        # Create client stats if they exist
        for client_stat in experiment.client_stats:
            client_stat.experiment_id = str(result.inserted_id)
            self.client_stat_store.create_or_update(client_stat)

        return str(result.inserted_id)

    @override
    def get(self, id: str, use_typing: bool = False) -> Experiment:
        """Get an experiment by id with its associated client statistics"""
        experiment = self.collection.find_one({"_id": pymongo.ObjectId(id)})
        if not experiment:
            return None

        # Get associated client statistics
        client_stats = self.client_stat_store.list_by_experiment(
            str(experiment["_id"]),
            use_typing=True
        )["result"]

        if use_typing:
            experiment["client_stats"] = client_stats
            return Experiment.from_dict(experiment)

        experiment["client_stats"] = [stat.to_dict() for stat in client_stats]
        return experiment

    def get_latest(self) -> Optional[Experiment]:
        """Get latest experiment, if any"""
        experiments = self.list(limit=1, skip=0, sort_key="timestamp")

        if len(experiments) == 0:
            return None
        else:
            return experiments[0]


experiment_store = ExperimentStore(mdb, "local.experiments")
