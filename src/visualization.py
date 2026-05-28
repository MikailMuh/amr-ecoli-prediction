"""Plot generation for results dashboard."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_per_antibiotic_auc(results: pd.DataFrame, ax=None):
    """Bar chart of AUC per antibiotic across models."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    pivot = results.pivot(index="antibiotic", columns="model", values="AUC")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Antibiotic")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_resistance_probability(predictions: pd.DataFrame, ax=None):
    """Horizontal bar chart of predicted resistance probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sorted_df = predictions.sort_values("probability", ascending=True)
    colors = ["#d62728" if p > 0.5 else "#2ca02c" for p in sorted_df["probability"]]
    ax.barh(sorted_df["antibiotic"], sorted_df["probability"], color=colors)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=0.5)
    ax.set_xlabel("Resistance Probability")
    ax.set_xlim(0, 1)
    return ax