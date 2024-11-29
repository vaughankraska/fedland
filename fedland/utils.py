import os
import json
import pandas as pd
from typing import Tuple, Optional
from fedland.database_models.experiment import Experiment
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_experiment_description(experiment_id: str, results_path: str = "results") -> Optional[str]:
    with open(f"{results_path}/experiments.json", mode="r") as ef:
        experiments = json.load(ef)

    for e in experiments:
        if e["id"] == experiment_id:
            return e["description"]

    return None


def plot_results_overview(df, results_path: str = "results", errorbars: Optional = ("ci", 90)):
    """
    Create consistent visualization of federated learning experiments.

    Default 90% CI bands around training stats
    """
    experiment_ids = df["experiment_id"].unique()
    n_experiments = len(experiment_ids)
    fig, axs = plt.subplots(n_experiments, 2, figsize=(16, 5*n_experiments))

    for i, experiment_id in enumerate(experiment_ids):
        exp_data = df[df["experiment_id"] == experiment_id]
        exp_description = get_experiment_description(experiment_id, results_path)
        # First subplot: Path Norms
        ax1 = axs[i, 0]
        ax1_twin = ax1.twinx()
        # Path Norm plot
        sns.lineplot(data=exp_data, x="iteration", y="path_norm",
                     hue="client_index", ax=ax1, palette="viridis")
        # Global Path Norm plot
        sns.lineplot(data=exp_data, x="iteration", y="global_path_norm",
                     hue="client_index", ax=ax1_twin, palette="viridis",
                     # label="Global",
                     linestyle="--")
        ax1.set_title(f"Path Norms - {exp_description}")
        ax1.set_xlabel("Training Iterations")
        ax1.set_ylabel("Path Norm")
        ax1_twin.set_ylabel("Global Path Norm")

        # Second subplot: Accuracies and Losses
        ax2 = axs[i, 1]
        # ax2_accuracy = ax2.twinx()
        ax2_loss = ax2.twinx()
        # Accuracy lines
        sns.lineplot(data=exp_data, x="iteration", y="train_accuracy",
                     ax=ax2, color="blue", label="Train Accuracy",
                     errorbar=errorbars)
        sns.lineplot(data=exp_data, x="iteration", y="test_accuracy",
                     ax=ax2, color="cyan", label="Test Accuracy",
                     errorbar=errorbars)
        # Loss lines
        sns.lineplot(data=exp_data, x="iteration", y="train_loss",
                     ax=ax2_loss, color="green", label="Train Loss",
                     errorbar=errorbars)
        sns.lineplot(data=exp_data, x="iteration", y="test_loss",
                     ax=ax2_loss, color="lime", label="Test Loss",
                     errorbar=errorbars)
        # Set y-axis limits
        ax2.set_ylim(0, 100)
        # loss_max = max(exp_data["train_loss"].max(), exp_data["test_loss"].max())
        # loss_min = max(exp_data["train_loss"].min(), exp_data["test_loss"].min())
        # print(f"lm: {loss_min}, lax: {loss_max}")
        # ax2_loss.set_ylim(0, loss_max)

        ax2.set_ylabel("Accuracy (%)", color="blue")
        ax2_loss.set_ylabel("Loss", color="green")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2_loss.tick_params(axis="y", labelcolor="green")

        ax2.set_title(f"Training Metrics - {exp_description}")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_loss.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
        ax2.legend(all_lines, all_labels, loc="center left", bbox_to_anchor=(1, 1))
        ax2_loss.get_legend().remove()

    plt.tight_layout()
    plt.show()
