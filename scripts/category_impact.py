import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import (
    CATEGORY_DISPLAY,
    EXPECTED_MODELS,
    TEMPORAL_DIMENSIONS,
    VISUAL_QUALITY_DIMENSIONS,
    SEMANTIC_DIMENSIONS,
    setup_plot_style,
    build_combined_evaluation_results,
    FIGURES_DIR,
    TABLES_DIR,
    MODEL_DISPLAY,
    DIMENSION_DISPLAY,
    MODEL_COLORS,
)


def prepare_category_impact_stats(df):
    """Aprēķina vidējās vērtības katrai (modelis, kategorija) kombinācijai."""
    subset = df[df["model"].isin(EXPECTED_MODELS)].copy()
    if "imaging_quality" in subset.columns and subset["imaging_quality"].max() > 1.5:
        subset["imaging_quality"] = subset["imaging_quality"] / 100

    all_metrics = TEMPORAL_DIMENSIONS + VISUAL_QUALITY_DIMENSIONS + SEMANTIC_DIMENSIONS
    means = subset.groupby(["model", "category"])[all_metrics].mean().reset_index()

    categories_sorted = sorted(subset["category"].unique())
    models_sorted = sorted(subset["model"].unique())

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    means.round(4).to_csv(TABLES_DIR / "category_impact.csv", index=False)

    return means, models_sorted, categories_sorted


def _plot_metric_group_by_category(
    means,
    models_sorted,
    categories_sorted,
    metrics,
    suptitle,
    filename,
    nrows,
    ncols,
    figsize,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # type: ignore

    handles_for_legend = []
    labels_for_legend = []

    n_models = len(models_sorted)
    bar_width = 0.8 / n_models
    x = np.arange(len(categories_sorted))
    x_labels = [CATEGORY_DISPLAY.get(c, c) for c in categories_sorted]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, model in enumerate(models_sorted):
            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_DISPLAY.get(model, model)

            model_data = means[means["model"] == model].set_index("category")
            values = [model_data.loc[c, metric] for c in categories_sorted]

            offset = (j - (n_models - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                color=color,
                label=label,
                edgecolor="black",
                linewidth=0.5,
            )

            if i == 0:
                handles_for_legend.append(bars[0])
                labels_for_legend.append(label)

        ax.set_title(DIMENSION_DISPLAY.get(metric, metric), fontsize=11)  # type: ignore
        ax.set_ylabel("Vidējais rādītājs", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)  # type: ignore
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

        values_all = means[metric].values
        vmin, vmax = values_all.min(), values_all.max()
        spread = vmax - vmin
        if spread < 1e-6:
            pad = max(vmax * 0.05, 0.01)
            ax.set_ylim(vmax - pad, vmax + pad)
        else:
            pad_low = spread * 0.15
            pad_high = spread * 0.15
            ax.set_ylim(max(0.0, vmin - pad_low), min(1.0, vmax + pad_high))

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(models_sorted),
        fontsize=10,
        title="Modelis",
        title_fontsize=11,
        frameon=True,
    )

    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_temporal_consistency(means, models_sorted, categories_sorted):
    _plot_metric_group_by_category(
        means=means,
        models_sorted=models_sorted,
        categories_sorted=categories_sorted,
        metrics=TEMPORAL_DIMENSIONS,
        suptitle="Kategoriju ietekme uz temporālo konsekvenci",
        filename="category_temporal_consistency.png",
        nrows=2,
        ncols=2,
        figsize=(14, 9),
    )


def plot_category_visual_quality(means, models_sorted, categories_sorted):
    _plot_metric_group_by_category(
        means=means,
        models_sorted=models_sorted,
        categories_sorted=categories_sorted,
        metrics=VISUAL_QUALITY_DIMENSIONS,
        suptitle="Kategoriju ietekme uz vizuālo kvalitāti",
        filename="category_visual_quality.png",
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )


def plot_category_semantic_alignment(means, models_sorted, categories_sorted):
    _plot_metric_group_by_category(
        means=means,
        models_sorted=models_sorted,
        categories_sorted=categories_sorted,
        metrics=SEMANTIC_DIMENSIONS,
        suptitle="Kategoriju ietekme uz semantisko atbilstību",
        filename="category_semantic_alignment.png",
        nrows=1,
        ncols=1,
        figsize=(9, 5),
    )


def main():
    setup_plot_style()

    df = build_combined_evaluation_results()

    if df is None or df.empty:
        print("Error: No data available to plot.")
        return

    means, models_sorted, categories_sorted = prepare_category_impact_stats(df)
    plot_category_temporal_consistency(means, models_sorted, categories_sorted)
    plot_category_visual_quality(means, models_sorted, categories_sorted)
    plot_category_semantic_alignment(means, models_sorted, categories_sorted)

    print("Category impact plots generated successfully in the figures directory.")


if __name__ == "__main__":
    main()
