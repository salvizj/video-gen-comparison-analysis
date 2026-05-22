from matplotlib import pyplot as plt
import pandas as pd

from base import (
    DIMENSION_DISPLAY,
    EXPECTED_MODELS,
    FIGURES_DIR,
    MODEL_COLORS,
    MODEL_DISPLAY,
    SEMANTIC_DIMENSIONS,
    TABLES_DIR,
    TEMPORAL_DIMENSIONS,
    VARIATION_DISPLAY,
    VARIATION_ORDER,
    VISUAL_QUALITY_DIMENSIONS,
    build_combined_evaluation_results,
    setup_plot_style,
)


def prepare_variation_impact_stats(df):
    subset = df[df["model"].isin(EXPECTED_MODELS)].copy()
    if "imaging_quality" in subset.columns and subset["imaging_quality"].max() > 1.5:
        subset["imaging_quality"] = subset["imaging_quality"] / 100

    all_metrics = TEMPORAL_DIMENSIONS + VISUAL_QUALITY_DIMENSIONS + SEMANTIC_DIMENSIONS
    means = subset.groupby(["model", "variation"])[all_metrics].mean().reset_index()
    means["variation"] = pd.Categorical(
        means["variation"], categories=VARIATION_ORDER, ordered=True
    )
    means = means.sort_values(["model", "variation"]).reset_index(drop=True)

    models_sorted = sorted(subset["model"].unique())

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    means.round(4).to_csv(TABLES_DIR / "variation_impact.csv", index=False)

    return means, models_sorted


def _plot_metric_group_by_variation(
    means,
    models_sorted,
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

    x_labels = [VARIATION_DISPLAY.get(v, v) for v in VARIATION_ORDER]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for model in models_sorted:
            model_data = means[means["model"] == model].sort_values("variation")
            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_DISPLAY.get(model, model)

            (line,) = ax.plot(
                x_labels,  # type: ignore
                model_data[metric],
                marker="o",
                markersize=7,
                linewidth=2,
                color=color,
                label=label,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

            for x, y in zip(x_labels, model_data[metric]):
                ax.annotate(
                    f"{y:.3f}",
                    (x, y),  # type: ignore
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7,
                    color=color,
                )

            if i == 0:
                handles_for_legend.append(line)
                labels_for_legend.append(label)

        ax.set_title(DIMENSION_DISPLAY.get(metric, metric), fontsize=11)  # type: ignore
        ax.set_xlabel("Ievadnes sarežģītība", fontsize=9)
        ax.set_ylabel("Vidējais rādītājs", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        values = means[metric].values
        vmin, vmax = values.min(), values.max()
        spread = vmax - vmin
        if spread < 1e-6:
            pad = max(vmax * 0.05, 0.01)
            ax.set_ylim(vmax - pad, vmax + pad)
        else:
            pad_low = spread * 0.20
            pad_high = spread * 0.35
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


def plot_variation_temporal_consistency(means, models_sorted):
    _plot_metric_group_by_variation(
        means=means,
        models_sorted=models_sorted,
        metrics=TEMPORAL_DIMENSIONS,
        suptitle="Ievadnes sarežģītības ietekme uz temporālo konsekvenci",
        filename="variation_temporal_consistency.png",
        nrows=2,
        ncols=2,
        figsize=(11, 8),
    )


def plot_variation_visual_quality(means, models_sorted):
    _plot_metric_group_by_variation(
        means=means,
        models_sorted=models_sorted,
        metrics=VISUAL_QUALITY_DIMENSIONS,
        suptitle="Ievadnes sarežģītības ietekme uz vizuālo kvalitāti",
        filename="variation_visual_quality.png",
        nrows=1,
        ncols=2,
        figsize=(11, 4.5),
    )


def plot_variation_semantic_alignment(means, models_sorted):
    _plot_metric_group_by_variation(
        means=means,
        models_sorted=models_sorted,
        metrics=SEMANTIC_DIMENSIONS,
        suptitle="Ievadnes sarežģītības ietekme uz semantisko atbilstību",
        filename="variation_semantic_alignment.png",
        nrows=1,
        ncols=1,
        figsize=(6.5, 4.5),
    )


def main():
    setup_plot_style()

    df = build_combined_evaluation_results()

    if df is None or df.empty:
        print("Error: No data available to plot.")
        return

    means, models_sorted = prepare_variation_impact_stats(df)
    plot_variation_temporal_consistency(means, models_sorted)
    plot_variation_visual_quality(means, models_sorted)
    plot_variation_semantic_alignment(means, models_sorted)

    print("Variation impact plots generated successfully in the figures directory.")


if __name__ == "__main__":
    main()
