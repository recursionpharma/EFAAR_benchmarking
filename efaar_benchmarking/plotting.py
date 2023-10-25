import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_recall(df):
    """
    Plots a line plot of recall values for several thereshold pairs for each source.

    Parameters:
    df (pandas.DataFrame): A dataframe with "source" column and "recall_X_Y" columns for several [X, Y] pairs.

    Returns:
    None
    """
    xy_pairs = [col.split("_")[1:] for col in df.columns if col.startswith("recall_")]

    col_cnt = 5
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(nrows=round(len(df['source'].unique())/col_cnt), ncols=col_cnt, figsize=(15, 3), squeeze=False)

    color = "r"
    # Plot each source as a separate subplot
    for i, source in enumerate(df['source'].unique()):
        source_df = df[df['source'] == source]
        x_values_orig = [f"{x}, {y}" for x, y in xy_pairs]
        y_values = [list(source_df[f"recall_{x}_{y}"]) for x, y in xy_pairs]
        x_values = [i for i, sublist in zip(x_values_orig, y_values) for _ in sublist]
        tmp_data = pd.DataFrame({'x': x_values, 'y': sum(y_values, [])})
        curr_ax = axs[i//col_cnt, i%col_cnt]
        sns.lineplot(ax=curr_ax, x="x", y="y", data=tmp_data, color=color, marker="o", markersize=8, markerfacecolor=color, markeredgewidth=2, errorbar=('ci', 99))
        curr_ax.set_title(source)
        curr_ax.set_xlabel("Recall Thresholds")
        curr_ax.set_ylabel("Recall Value")
        curr_ax.set_xticks(range(len(x_values_orig)))
        curr_ax.set_xticklabels(x_values_orig, rotation=45)

    plt.tight_layout()
    plt.show()
