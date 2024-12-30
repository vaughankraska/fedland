import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from fedland.database_models.experiment import Experiment
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_experiment_data(
    path: str, ignore_validate=False
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
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

                df_training = pd.read_json(f"{root}/{dir}/training.json")

                df_training["client_index"] = client["client_index"]
                df_training["experiment_id"] = client["experiment_id"]
                df_training["pct_change_path_norm"] = df_training[
                    "path_norm"
                ].pct_change()
                df_training["pct_change_global_path_norm"] = df_training[
                    "global_path_norm"
                ].pct_change()

                df_training_results = pd.concat([df_training_results, df_training])

                if not ignore_validate:
                    df_validate = pd.read_json(f"{root}/{dir}/validate.json")
                    df_validate_results = pd.concat([df_validate_results, df_validate])

                clients_data.append(client)
            except Exception as e:
                print(f"Error reading dir {dir}: {e}")

    return df_training_results, df_validate_results, clients_data


def load_all_training_results(
    results_path: str = "results", ignore_validate=False
) -> pd.DataFrame:
    with open(f"{results_path}/experiments.json", mode="r") as ef:
        experiments = json.load(ef)

    df_all = pd.DataFrame()
    for exp_dict in experiments:
        experiment = Experiment.from_dict(exp_dict)
        exp_path = f"{results_path}/{experiment.id}"

        if not os.path.exists(exp_path):
            print(f"WARN: Experiment id '{exp_path}' not found")
        else:
            df_experiment, _, _ = read_experiment_data(exp_path, ignore_validate)
            df_all = pd.concat([df_all, df_experiment])

    # Create iteration variable for each experiment's clients
    df_all = df_all.sort_values(by=["experiment_id", "client_index", "timestamp"])
    df_all['iteration'] = df_all.groupby(['experiment_id', 'client_index']).cumcount() + 1
    df_all = df_all.reset_index()

    return df_all


def get_experiment_description(
    experiment_id: str, results_path: str = "results"
) -> Optional[str]:
    with open(f"{results_path}/experiments.json", mode="r") as ef:
        experiments = json.load(ef)

    for e in experiments:
        if e["id"] == experiment_id:
            return e["description"]

    return None

def summarise_experiments(df, results_path = './results'):
    '''
    Create a dataframe that contains (experiment, client)-wise summarised statistics, including
        Final Local Path-norm
        Final Testing Loss
        Final Testing Accuracy
        Correlation Local Path-norm & Training Loss
        Correlation Local Path-norm & Training Accuracy
        Correlation Local Path-norm & Testing Loss
        Correlation Local Path-norm & Testing Accuracy
        Average Percentage Change of Local Path-norm 1 Round before Aggregation
        Average Percentage Change of Local Path-norm on the Aggregation Round
        Average Percentage Change of Local Path-norm 1 Round after Aggregation
        Average Percentage Change of Local Path-norm on Non-aggregation Rounds
    '''
    df_summary = pd.DataFrame(columns = ['experiment_id',
                                         'description',
                                         'client_index',
                                         'final_path_norm',
                                         'final_test_loss',
                                         'final_test_acc',
                                         'corr_path_norm_train_loss',
                                         'corr_path_norm_train_acc',
                                         'corr_path_norm_test_loss',
                                         'corr_path_norm_test_acc',
                                         'avg_pct_change_path_norm_before_aggregation_rounds',
                                         'avg_pct_change_path_norm_aggregation_rounds',
                                         'avg_pct_change_path_norm_after_aggregation_rounds',
                                         'avg_pct_change_path_norm_non_aggregation_rounds'])
    df_nested = pd.DataFrame()
    
    for experiment in df['experiment_id'].unique():
        description = get_experiment_description(experiment)
        exp_path = f"{results_path}/{experiment}"
        _, _, clients_data = read_experiment_data(exp_path, ignore_validate = True)
        
        df_nested_experiment = pd.DataFrame(clients_data)
        df_nested_experiment["total_data"] = df_nested_experiment["data_indices"].apply(lambda x: len(x))
        df_nested = pd.concat([df_nested, df_nested_experiment], ignore_index = True)
            
        df_experiment = df[df['experiment_id'] == experiment]
        for client in df_experiment['client_index'].unique():
            df_experiment_client = df_experiment[df_experiment['client_index'] == client].reset_index(drop = True)
    
            pct_change_path_norm_before_aggregation_rounds = df_experiment_client['pct_change_path_norm'][df_experiment_client['epoch'] == df_experiment_client['epoch'].max()].mean()
            pct_change_path_norm_aggregation_rounds = df_experiment_client['pct_change_path_norm'][df_experiment_client['epoch'] == 0].mean()
            pct_change_path_norm_after_aggregation_rounds = df_experiment_client['pct_change_path_norm'][df_experiment_client['epoch'] == 1].mean()
            pct_change_path_norm_non_aggregation_rounds = df_experiment_client['pct_change_path_norm'][df_experiment_client['epoch'] != 0].mean()
            
            df_experiment_client_summary = pd.DataFrame([[experiment,
                                                          description,
                                                          client,
                                                          df_experiment_client['path_norm'].iloc[-1],
                                                          df_experiment_client['test_loss'].iloc[-1],
                                                          df_experiment_client['test_accuracy'].iloc[-1],
                                                          df_experiment_client['path_norm'].corr(df_experiment_client['train_loss']),
                                                          df_experiment_client['path_norm'].corr(df_experiment_client['train_accuracy']),
                                                          df_experiment_client['path_norm'].corr(df_experiment_client['test_loss']),
                                                          df_experiment_client['path_norm'].corr(df_experiment_client['test_accuracy']),
                                                          pct_change_path_norm_before_aggregation_rounds,
                                                          pct_change_path_norm_aggregation_rounds,
                                                          pct_change_path_norm_after_aggregation_rounds,
                                                          pct_change_path_norm_non_aggregation_rounds]],
                                                        columns = ['experiment_id',
                                                                   'description',
                                                                   'client_index',
                                                                   'final_path_norm',
                                                                   'final_test_loss',
                                                                   'final_test_acc',
                                                                   'corr_path_norm_train_loss',
                                                                   'corr_path_norm_train_acc',
                                                                   'corr_path_norm_test_loss',
                                                                   'corr_path_norm_test_acc',
                                                                   'avg_pct_change_path_norm_before_aggregation_rounds',
                                                                   'avg_pct_change_path_norm_aggregation_rounds',
                                                                   'avg_pct_change_path_norm_after_aggregation_rounds',
                                                                   'avg_pct_change_path_norm_non_aggregation_rounds'])
            df_summary = pd.concat([df_summary, df_experiment_client_summary], ignore_index = True)

    df_summary = df_summary.merge(df_nested[['experiment_id', 'client_index', 'total_data']], on = ['experiment_id', 'client_index'], validate = '1:1')
    df_summary.to_csv('./results/summary.csv')
    print('Summary saved to local file system')

    return df_summary

def plot_client_info(
        experiment_id: str,
        results_path: str = "results",
        figsize: tuple = (12, 10)
        ):
    """
    Plots each client's data proportions and class distribution for an experiment.
    """
    exp_path = f"{results_path}/{experiment_id}"
    _, _, clients_data = read_experiment_data(exp_path, ignore_validate=True)

    if not clients_data:
        raise ValueError(f"No client data found for experiment {experiment_id}")

    experiment_desc = get_experiment_description(experiment_id=experiment_id, results_path=results_path)
    class_keys = sorted(clients_data[0]['balance']['class_frequencies'].keys())

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Client Info for "{experiment_desc}"', fontsize=12)

    df_nested = pd.DataFrame(clients_data)
    df_nested["total_data"] = df_nested["data_indices"].apply(lambda x: len(x))

    class_frequencies = []
    client_labels = []

    for client in sorted(clients_data, key=lambda client: client["client_index"]):
        class_frequencies.append([client['balance']['class_frequencies'].get(key, 0) for key in class_keys])
        client_labels.append(client["client_index"])

    # Plot total data count per client as a bar plot
    sns.barplot(data=df_nested, x="client_index", y="total_data", ax=ax1)

    ax1.set_title("Total Data Count per Client")
    ax1.set_xlabel("Client Index")
    ax1.set_ylabel("Total Number of Samples")

    # Plot frequencies heatmap
    sns.heatmap(class_frequencies,
                annot=True,
                fmt='.3f',
                cmap='viridis',
                xticklabels=class_keys,
                yticklabels=[f'Client {c}' for c in client_labels],
                vmin=0,
                vmax=0.75,
                ax=ax2)
    ax2.set_title("Class Frequencies per Client")
    ax2.set_xlabel("Classes")
    ax2.set_ylabel("Clients")
    plt.tight_layout()
    plt.show()


def plot_results_overview(
        df,
        results_path: str = "results",
        errorbars: Optional = ("ci", 90),
        epoch: Optional[int] = None,
        pct_change: bool = False,
        iter_window: Optional[Tuple[int, int]] = None
):
    """
    Create consistent visualization of federated learning experiments.

    Default 90% CI bands around training stats
    """
    experiment_ids = df["experiment_id"].unique()
    n_experiments = len(experiment_ids)
    fig, axs = plt.subplots(n_experiments, 2, figsize=(16, 5 * n_experiments))

    for i, experiment_id in enumerate(experiment_ids):
        exp_data = df[df["experiment_id"] == experiment_id]
        exp_description = get_experiment_description(experiment_id, results_path)
        pre_title = ""

        if epoch is not None:
            exp_data = exp_data[exp_data["epoch"] == epoch]
            pre_title = pre_title + f"Epoch=={epoch}"

        if pct_change:
            exp_data.loc[:, "path_norm"] = exp_data["path_norm"].pct_change()
            exp_data.loc[:, "global_path_norm"] = exp_data["global_path_norm"].pct_change()
            exp_data.loc[exp_data["global_path_norm"] == 0, "global_path_norm"] = None
            exp_data.loc[:, "global_path_norm"] = exp_data["global_path_norm"].ffill()

            exp_data = exp_data.dropna()
            pre_title = pre_title + " %Ch."

        if iter_window is not None:
            if iter_window[0] > iter_window[1]:
                raise ValueError("Lower bound greater than upper boud on iter_window")
            exp_data = exp_data[exp_data["iteration"] >= iter_window[0]]
            exp_data = exp_data[exp_data["iteration"] <= iter_window[1]]

        # First subplot: Path Norms
        ax1 = axs[i, 0]
        path_norm_min = min(exp_data["path_norm"].min(), exp_data["global_path_norm"].min())
        path_norm_max = max(exp_data["path_norm"].max(), exp_data["global_path_norm"].max())

        # Path Norm plot
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="path_norm",
            hue="client_index",
            ax=ax1,
            palette="viridis",
        )
        # Global Path Norm plot
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="global_path_norm",
            hue="client_index",
            ax=ax1,
            palette="viridis",
            linestyle="--",
        )
        ax1.set_title(f"{pre_title} Path Norms - {exp_description}")
        ax1.set_ylim(path_norm_min, path_norm_max)
        ax1.set_xlabel("Training Iterations")

        if pct_change:
            ax1.set_ylabel("% Change - Path Norm")
        else:
            ax1.set_ylabel("Path Norm")

        # Second subplot: Accuracies and Losses
        ax2 = axs[i, 1]
        # ax2_accuracy = ax2.twinx()
        ax2_loss = ax2.twinx()
        # Accuracy lines
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="train_accuracy",
            ax=ax2,
            color="blue",
            label="Train Accuracy",
            errorbar=errorbars,
        )
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="test_accuracy",
            ax=ax2,
            color="cyan",
            label="Test Accuracy",
            errorbar=errorbars,
        )
        # Loss lines
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="train_loss",
            ax=ax2_loss,
            color="green",
            label="Train Loss",
            errorbar=errorbars,
        )
        sns.lineplot(
            data=exp_data,
            x="iteration",
            y="test_loss",
            ax=ax2_loss,
            color="lime",
            label="Test Loss",
            errorbar=errorbars,
        )

        # Vertical lines on epoch 0 (after aggregation)
        for epoch_val in exp_data[exp_data["epoch"] == 0]["iteration"]:
            ax1.axvline(x=epoch_val, color="gray", linestyle="--", alpha=0.01)
            ax2.axvline(x=epoch_val, color="gray", linestyle="--", alpha=0.01)

        # Set y-axis limits
        ax2.set_ylim(0, 100)

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


def plot_results_overview_plotly(
    df,
    n_clients,
    result_path="./results/",
    metrics=[
        "Path-norm",
        "Path-norm Percentage Change",
        "Training Accuracy",
        "Training Loss",
        "Testing Accuracy",
        "Testing Loss",
    ],
):
    """
    Create consistent visualization of federated learning experiments.
    Based on Plotly
    """

    # Define some useful colours
    colours = (
        px.colors.qualitative.Dark24[:2]
        + px.colors.qualitative.Dark24[6:8]
        + px.colors.qualitative.Dark24[14:]
        + [px.colors.qualitative.Dark24[5]]
    )

    experiment_ids = df["experiment_id"].unique()
    n_experiments = len(experiment_ids)
    n_clients_experiments = (
        df[["experiment_id", "client_index"]].groupby("experiment_id").nunique()
    )
    n_valid_experiments = (n_clients_experiments == n_clients).sum().iloc[0].item()

    row_count = 1

    fig = make_subplots(
        rows=n_valid_experiments,
        cols=len(metrics),
        row_titles=[get_experiment_description(i) for i in experiment_ids],
        subplot_titles=metrics * n_experiments,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for j in range(n_experiments):
        df_experiment = df[df["experiment_id"] == experiment_ids[j]].reset_index(drop = True)
        n_clients_experiment = len(df_experiment["client_index"].unique())

        if n_clients != n_clients_experiment:
            print(
                f"WARN: Experiment id '{experiment_ids[j]}' does not have the same number of clients.\nThe results are not plotted"
            )
            continue

        # Local Path-norm
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "path_norm"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=True if j == 0 else False,
                    ),
                    row=row_count,
                    col=1,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=1),
                fig.update_yaxes(title_text="Path-norm", row=row_count, col=1),
            ]
            for i in df_experiment["client_index"].unique()
        ]
        # Global Path-norm
        fig.add_trace(
            go.Scatter(
                x=df_experiment[df_experiment["client_index"] == 0]['iteration'] + 1,
                y=df_experiment[df_experiment["client_index"] == 0]["global_path_norm"],
                mode="lines",
                legendgroup="Global",
                line=dict(dash="dot"),
                line_color=colours[n_clients % len(colours)],
                name="Global",
                showlegend=True if j == 0 else False,
            ),
            row=row_count,
            col=1,
        )
        # Local Path-norm Percentage Change
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "pct_change_path_norm"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=False,
                    ),
                    row=row_count,
                    col=2,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=2),
                fig.update_yaxes(
                    title_text="Path-norm Percentage Change", row=row_count, col=2
                ),
            ]
            for i in df_experiment["client_index"].unique()
        ]
        # Global Path-norm Percentage Change
        fig.add_trace(
            go.Scatter(
                x=df_experiment[
                    (df_experiment["client_index"] == 0)
                    & (df_experiment["pct_change_global_path_norm"] != 0)
                ].index
                + 1,
                y=df_experiment[
                    (df_experiment["client_index"] == 0)
                    & (df_experiment["pct_change_global_path_norm"] != 0)
                ]["pct_change_global_path_norm"],
                mode="lines",
                legendgroup="Global",
                line=dict(dash="dot"),
                line_color=colours[n_clients % len(colours)],
                name="Global",
                showlegend=False,
            ),
            row=row_count,
            col=2,
        )
        # Training Accuracy
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "train_accuracy"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=False,
                    ),
                    row=row_count,
                    col=3,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=3),
                fig.update_yaxes(title_text="Training Accuracy", row=row_count, col=3),
            ]
            for i in df_experiment["client_index"].unique()
        ]
        # Training Loss
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "train_loss"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=False,
                    ),
                    row=row_count,
                    col=4,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=4),
                fig.update_yaxes(title_text="Training Loss", row=row_count, col=4),
            ]
            for i in df_experiment["client_index"].unique()
        ]
        # Testing Accuracy
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "test_accuracy"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=False,
                    ),
                    row=row_count,
                    col=5,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=5),
                fig.update_yaxes(title_text="Testing Accuracy", row=row_count, col=5),
            ]
            for i in df_experiment["client_index"].unique()
        ]
        # Testing Loss
        [
            [
                fig.add_trace(
                    go.Scatter(
                        x=df_experiment[df_experiment["client_index"] == i]['iteration'] + 1,
                        y=df_experiment[df_experiment["client_index"] == i][
                            "test_loss"
                        ],
                        mode="lines",
                        legendgroup="Client " + str(i),
                        line_color=colours[i % len(colours)],
                        name="Client " + str(i),
                        showlegend=False,
                    ),
                    row=row_count,
                    col=6,
                ),
                # Update x- and y-axes properties
                fig.update_xaxes(title_text="Iteration", row=row_count, col=6),
                fig.update_yaxes(title_text="Testing Loss", row=row_count, col=6),
            ]
            for i in df_experiment["client_index"].unique()
        ]

        row_count += 1

    fig.update_layout(
        height=900 * n_valid_experiments,
        width=900 * len(metrics),
        legend=dict(orientation="h"),
        title_text=str(n_clients) + "-client Experiment Results",
    )
    fig.update_xaxes(range=[df['iteration'].min() - 5, df['iteration'].max() + 5])

    fig.show()
    fig.write_image(
        result_path + "overview.png",
        height=900 * n_valid_experiments,
        width=900 * len(metrics),
    )


def plot_results_evolution_plotly(df, result_path="./results/"):
    """
    Create consistent visualization of federated learning experiments.
    Based on Plotly
    """
    # Define some useful colours
    colours = (
        px.colors.qualitative.Dark24[:2]
        + px.colors.qualitative.Dark24[6:8]
        + px.colors.qualitative.Dark24[14:]
        + [px.colors.qualitative.Dark24[5]]
    )

    experiment_ids = df["experiment_id"].unique()
    n_experiments = len(experiment_ids)

    df_global_path_norm = df[["experiment_id", "global_path_norm", "iteration"]].drop_duplicates(subset = ["experiment_id", "global_path_norm"]).reset_index(drop = True)
    df_global_path_norm["description"] = [
        get_experiment_description(i) for i in df_global_path_norm["experiment_id"]
    ]

    fig = go.Figure()
    [
        fig.add_trace(
            go.Scatter(
                x=df_global_path_norm[
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ]['iteration']
                + 1,
                y=np.log10(
                    df_global_path_norm[
                        df_global_path_norm["experiment_id"] == experiment_ids[i]
                    ]["global_path_norm"]
                ),
                mode="lines",
                legendgroup=df_global_path_norm["description"][
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ].unique()[0],
                line_color=colours[i % len(colours)],
                name=df_global_path_norm["description"][
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ].unique()[0],
                showlegend=True,
            )
        )
        for i in range(n_experiments)
    ]
    fig.update_layout(
        height=900,
        width=1600,
        legend=dict(orientation="h"),
        title_text="Path-norm Evolution",
        xaxis_title="Training Iteration",
        yaxis_title="Log(Global Path-norm)",
    )
    fig.update_xaxes(
        range=[df_global_path_norm['iteration'].min() - 5, df_global_path_norm['iteration'].max() + 5]
    )

    fig.show()
    fig.write_image(result_path + "evolution.png", height=900, width=1600)


def plot_results_pct_change_plotly(df, result_path="./results/"):
    """
    Create consistent visualization of federated learning experiments.
    Based on Plotly
    """
    # Define some useful colours
    colours = (
        px.colors.qualitative.Dark24[:2]
        + px.colors.qualitative.Dark24[6:8]
        + px.colors.qualitative.Dark24[14:]
        + [px.colors.qualitative.Dark24[5]]
    )

    experiment_ids = df["experiment_id"].unique()
    n_experiments = len(experiment_ids)

    df_global_path_norm = df[df["pct_change_global_path_norm"] != 0][
        ["experiment_id", "pct_change_global_path_norm", "iteration"]
    ].drop_duplicates(subset = ["experiment_id", "pct_change_global_path_norm"]).reset_index(drop = True)
    df_global_path_norm["description"] = [
        get_experiment_description(i) for i in df_global_path_norm["experiment_id"]
    ]

    fig = go.Figure()
    [
        fig.add_trace(
            go.Scatter(
                x=df_global_path_norm[
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ]['iteration']
                + 1,
                y=df_global_path_norm[
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ]["pct_change_global_path_norm"],
                mode="lines",
                legendgroup=df_global_path_norm["description"][
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ].unique()[0],
                line_color=colours[i % len(colours)],
                name=df_global_path_norm["description"][
                    df_global_path_norm["experiment_id"] == experiment_ids[i]
                ].unique()[0],
                showlegend=True,
            )
        )
        for i in range(n_experiments)
    ]
    fig.update_layout(
        height=900,
        width=1600,
        legend=dict(orientation="h"),
        title_text="Path-norm Percantage Change",
        xaxis_title="Training Iteration",
        yaxis_title="Global Path-norm Percantage Change",
    )
    fig.update_xaxes(
        range=[df_global_path_norm['iteration'].min() - 5, df_global_path_norm['iteration'].max() + 5]
    )

    fig.show()
    fig.write_image(result_path + "pct_change.png", height=900, width=1600)

def plot_final_round_plotly(df_summary, result_path="./results/"):
    # Define some useful colours
    colours = (
        px.colors.qualitative.Dark24[:2]
        + px.colors.qualitative.Dark24[6:8]
        + px.colors.qualitative.Dark24[14:]
        + [px.colors.qualitative.Dark24[5]]
    )
    
    experiment_ids = df_summary["experiment_id"].unique()
    n_experiments = len(experiment_ids)
    
    fig = make_subplots(rows=1,
                        cols=3,
                        subplot_titles=['Final Path-norm', 'Final Test Loss', 'Final Test Accuracy'],
                        horizontal_spacing=0.05)
    for i in range(n_experiments):
        df_summary_experiment = df_summary[(df_summary['experiment_id'] == experiment_ids[i])]
        
        total_data = df_summary_experiment['total_data'].sum()
        final_path_norm_min_total_data = df_summary_experiment[df_summary_experiment['total_data'] == df_summary_experiment['total_data'].min()]['final_path_norm']
        final_test_loss_min_total_data = df_summary_experiment[df_summary_experiment['total_data'] == df_summary_experiment['total_data'].min()]['final_test_loss']
        final_test_accuracy_min_total_data = df_summary_experiment[df_summary_experiment['total_data'] == df_summary_experiment['total_data'].min()]['final_test_acc']
        
        if len(final_path_norm_min_total_data) == 1:
            final_path_norm_min_total_data = final_path_norm_min_total_data.iloc[0]
            final_test_loss_min_total_data = final_test_loss_min_total_data.iloc[0]
            final_test_accuracy_min_total_data = final_test_accuracy_min_total_data.iloc[0]
    
            fig.add_trace(
                go.Scatter(
                    x=df_summary_experiment['total_data'] / total_data,
                    y=(df_summary_experiment['final_path_norm'] - final_path_norm_min_total_data) / final_path_norm_min_total_data,
                    legendgroup=df_summary_experiment['description'].unique()[0],
                    line_color=colours[i % len(colours)],
                    name=df_summary_experiment["description"].unique()[0],
                    showlegend=True,
                ),
                row=1,
                col=1
            )
            # Update x- and y-axes properties
            fig.update_xaxes(title_text="Proportion Number of Samples", row=1, col=1)
            fig.update_yaxes(title_text="Percentage Difference", row=1, col=1)
            
            fig.add_trace(
                go.Scatter(
                    x=df_summary_experiment['total_data'] / total_data,
                    y=(df_summary_experiment['final_test_loss'] - final_test_loss_min_total_data) / final_test_loss_min_total_data,
                    legendgroup=df_summary_experiment['description'].unique()[0],
                    line_color=colours[i % len(colours)],
                    name=df_summary_experiment["description"].unique()[0],
                    showlegend=False,
                ),
                row=1,
                col=2
            )
            # Update x- and y-axes properties
            fig.update_xaxes(title_text="Proportion Number of Samples", row=1, col=2)
            fig.update_yaxes(title_text="Percentage Difference", row=1, col=2)
            
            fig.add_trace(
                go.Scatter(
                    x=df_summary_experiment['total_data'] / total_data,
                    y=(df_summary_experiment['final_test_acc'] - final_test_accuracy_min_total_data) / final_test_accuracy_min_total_data,
                    legendgroup=df_summary_experiment['description'].unique()[0],
                    line_color=colours[i % len(colours)],
                    name=df_summary_experiment["description"].unique()[0],
                    showlegend=False,
                ),
                row=1,
                col=3
            )
            # Update x- and y-axes properties
            fig.update_xaxes(title_text="Proportion Number of Samples", row=1, col=3)
            fig.update_yaxes(title_text="Percentage Difference", row=1, col=3)
        else:
            print('Experiement ' + experiment_ids[i] + ' has more than 1 clients with the least number of samples and hence is ignored.')
        
    fig.update_layout(
        height=900,
        width=2700,
        legend=dict(orientation="h"),
        title_text="Final Round Figures vs Number of Samples<br><sup>Relative to client with least # samples</sup>"
    )
    fig.update_xaxes(tickformat = '.2%')
    fig.update_yaxes(tickformat = '.2%')
    
    fig.show()
    fig.write_image(result_path + "final_round.png", height=900, width=2700)