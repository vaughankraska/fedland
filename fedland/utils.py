import os
import json
import pandas as pd
from typing import Tuple
from fedland.database_models.experiment import Experiment


def read_experiment_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Read all data from an experiment directory into a pandas dataframe

    Returns:
        Tuple[training_df, validate_df, clients_list]
    """
    df_training_results = pd.DataFrame()
    df_validate_results = pd.DataFrame()
    clients_data = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            try:
                with open(f"{root}/{dir}/client.json", mode="r") as client_file:
                    client = json.load(client_file)

                df_validate = pd.read_json(f"{root}/{dir}/validate.json")

                df_training = pd.read_json(f"{root}/{dir}/training.json")
                df_training["client_index"] = client["client_index"]
                df_training["experiment_id"] = client["experiment_id"]

                df_training_results = pd.concat([df_training_results, df_training])
                df_validate_results = pd.concat([df_validate_results, df_validate])
                clients_data.append(client)
            except Exception as e:
                print(f"Error reading dir {dir}: {e}")

    return df_training_results, df_validate_results, clients_data



def load_all_training_results(results_path: str = "results") -> pd.DataFrame:

    with open(f"{results_path}/experiments.json", mode="r") as ef:
        experiments = json.load(ef)

    df_all = pd.DataFrame()
    for exp_dict in experiments:

        experiment = Experiment.from_dict(exp_dict)
        exp_path = f"{results_path}/{experiment.id}"

        if not os.path.exists(exp_path):
            print(f"WARN: Experiment id '{exp_path}' not found")
        else:
            df_experiment, _, _ = read_experiment_data(exp_path)
            df_all = pd.concat([df_all, df_experiment])

    return df_all
