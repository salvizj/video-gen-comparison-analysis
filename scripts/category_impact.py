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
)


def prepare_category_impact_stats(df):
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


def _plot_heatmap(
    means,
    models_sorted,
    categories_sorted,
    metrics,
    suptitle,
    filename,
    nrows=1,
    ncols=None,
):
    n_metrics = len(metrics)
    if ncols is None:
        ncols = n_metrics

    n_cats = len(categories_sorted)
    cell_width = 1.2
    cell_height = 0.9

    fig_w = ncols * (n_cats * cell_width + 1.5)
    fig_h = nrows * (len(models_sorted) * cell_height + 1.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    x_labels = [CATEGORY_DISPLAY.get(c, c) for c in categories_sorted]
    y_labels = [MODEL_DISPLAY.get(m, m) for m in models_sorted]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        data = np.array(
            [
                [
                    means[(means["model"] == m) & (means["category"] == c)][
                        metric
                    ].values[0]
                    for c in categories_sorted
                ]
                for m in models_sorted
            ]
        )

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xticks(range(len(categories_sorted)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(DIMENSION_DISPLAY.get(metric, metric), fontsize=11)

        for row in range(len(models_sorted)):
            for col in range(len(categories_sorted)):
                val = data[row, col]
                text_color = (
                    "black"
                    if 0.3 < (val - data.min()) / (data.max() - data.min() + 1e-9) < 0.7
                    else "white"
                )
                ax.text(
                    col,
                    row,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                    fontweight="bold",
                )

    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def main():
    setup_plot_style()
    df = build_combined_evaluation_results()

    if df is None or df.empty:
        print("Error: No data available.")
        return

    means, models_sorted, categories_sorted = prepare_category_impact_stats(df)

    _plot_heatmap(
        means,
        models_sorted,
        categories_sorted,
        metrics=TEMPORAL_DIMENSIONS,
        suptitle="Kategoriju ietekme uz temporālo konsekvenci",
        filename="category_temporal_consistency_heatmap.png",
        nrows=2,
        ncols=2,
    )

    _plot_heatmap(
        means,
        models_sorted,
        categories_sorted,
        metrics=VISUAL_QUALITY_DIMENSIONS,
        suptitle="Kategoriju ietekme uz vizuālo kvalitāti",
        filename="category_visual_quality_heatmap.png",
    )

    _plot_heatmap(
        means,
        models_sorted,
        categories_sorted,
        metrics=SEMANTIC_DIMENSIONS,
        suptitle="Kategoriju ietekme uz semantisko atbilstību",
        filename="category_semantic_alignment_heatmap.png",
    )


if __name__ == "__main__":
    main()
