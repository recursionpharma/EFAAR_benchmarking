import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_recall(df, color="red"):
    """
    Plots a line plot of recall values for several thereshold pairs for each source.

    Parameters:
    df (pandas.DataFrame): A dataframe with "source" column and "recall_X_Y" columns for several [X, Y] pairs.

    Returns:
    None
    """
    recall_thr_pairs = [col.split("_")[1:] for col in df.columns if col.startswith("recall_")]
    x_values = [f"{x}, {y}" for x, y in recall_thr_pairs]
    random_recall_values = [float(x) + 1 - float(y) for x, y in recall_thr_pairs]

    col_cnt = 5
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(
        nrows=math.ceil(len(df["source"].unique()) / col_cnt), ncols=col_cnt, figsize=(15, 3), squeeze=False
    )

    # Plot each source as a separate subplot
    for i, source in enumerate(df["source"].unique()):
        source_df = df[df["source"] == source]
        y_values = [list(source_df[f"recall_{x}_{y}"]) for x, y in recall_thr_pairs]
        # repeat x_values to match the length of y_values since we have multiple iterations of computations
        # if x_values=["0.01, 0.99", "0.05, 0.95", "0.1, 0.9"] and there are three iterations of computations,
        # then x_values_all=["0.01, 0.99", "0.01, 0.99", "0.01, 0.99", "0.05, 0.95", "0.05, 0.95", "0.05, 0.95",
        # "0.1, 0.9", "0.1, 0.9", "0.1, 0.9"]
        x_values_all = [i for i, sublist in zip(x_values, y_values) for _ in sublist]
        recall_data = pd.DataFrame({"x": x_values_all, "y": sum(y_values, [])})
        curr_ax = axs[i // col_cnt, i % col_cnt]
        sns.lineplot(
            ax=curr_ax,
            x="x",
            y="y",
            data=recall_data,
            color=color,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgewidth=2,
            errorbar=("ci", 95),
        )
        random_values = [i for i, sublist in zip(random_recall_values, y_values) for _ in sublist]
        random_data = pd.DataFrame({"x": x_values_all, "y": random_values})
        sns.lineplot(
            ax=curr_ax,
            x="x",
            y="y",
            data=random_data,
            color="gray",
            marker="o",
            markersize=5,
            markerfacecolor="gray",
            markeredgewidth=2,
            errorbar=("ci", 95),
        )
        curr_ax.set_title(source)
        curr_ax.set_xlabel("Recall Thresholds")
        curr_ax.set_ylabel("Recall Value")
        curr_ax.set_xticks(range(len(x_values)))
        curr_ax.set_xticklabels(x_values, rotation=45)

    plt.tight_layout()
    plt.show()
