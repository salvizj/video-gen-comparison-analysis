from matplotlib.container import BarContainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base import (
    EXPECTED_MODELS,
    LENGTH_ORDER,
    SEMANTIC_DIMENSIONS,
    TEMPORAL_DIMENSIONS,
    VISUAL_QUALITY_DIMENSIONS,
    setup_plot_style,
    build_combined_evaluation_results,
    FIGURES_DIR,
    TABLES_DIR,
    MODEL_DISPLAY,
    DIMENSION_DISPLAY,
    DIMENSION_ORDER,
    MODEL_COLORS,
)


def prepare_time_taken_data(df):
    plot_df = df[df["model"].isin(EXPECTED_MODELS)].copy()
    plot_df["model_display"] = (
        plot_df["model"].map(MODEL_DISPLAY).fillna(plot_df["model"])
    )

    avg_time_df = plot_df.groupby(["model", "model_display", "length"], as_index=False)[
        "time_seconds"
    ].mean()

    avg_time_df["length"] = pd.Categorical(
        avg_time_df["length"], categories=LENGTH_ORDER, ordered=True
    )
    avg_time_df = avg_time_df.sort_values(["model_display", "length"])

    model_order_df = (
        avg_time_df[["model", "model_display"]]
        .drop_duplicates()
        .sort_values("model_display")
    )
    return (
        avg_time_df,
        model_order_df["model_display"].tolist(),
        model_order_df["model"].tolist(),
    )


def plot_time_taken(avg_time_df, model_display_order, raw_model_order):
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        x="model_display",
        y="time_seconds",
        hue="length",
        data=avg_time_df,
        order=model_display_order,
        hue_order=LENGTH_ORDER,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    num_hues = len(LENGTH_ORDER)
    for i, bar in enumerate(ax.patches):
        model_idx = i // num_hues
        if model_idx < len(raw_model_order):
            raw_model_name = raw_model_order[model_idx]
            color = MODEL_COLORS.get(raw_model_name, "#808080")
            bar.set_facecolor(color)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                fmt="%.0fs",
                padding=3,
                fontsize=9,
            )

    ax.set_title("Vidējais video ģenerēšanas laiks", fontsize=14)
    ax.set_xlabel("Modelis", fontsize=11)
    ax.set_ylabel("Laiks (sekundes)", fontsize=11)
    ax.legend(title="Video garums", loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    ymax = avg_time_df["time_seconds"].max()
    ax.set_ylim(0, ymax * 1.15)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "time_taken.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def prepare_overall_evaluation_stats(df):
    subset = df[df["model"].isin(EXPECTED_MODELS)].copy()
    if "imaging_quality" in subset.columns and subset["imaging_quality"].max() > 1.5:
        subset["imaging_quality"] = subset["imaging_quality"] / 100

    means = subset.groupby("model")[DIMENSION_ORDER].mean()
    models_sorted = sorted(means.index)
    colors = [MODEL_COLORS.get(m, "gray") for m in models_sorted]
    labels = [MODEL_DISPLAY.get(m, m) for m in models_sorted]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    means.round(4).to_csv(TABLES_DIR / "overall_evaluation.csv")

    return means, models_sorted, colors, labels


def _plot_metric_group(
    means,
    models_sorted,
    colors,
    labels,
    metrics,
    suptitle,
    filename,
    nrows,
    ncols,
    figsize,
    legend_loc="lower right",
    legend_bbox=(0.98, 0.05),
    legend_ncol=2,
):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # type: ignore

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = means.loc[models_sorted, metric].values
        bars = ax.bar(
            range(len(models_sorted)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

        ax.set_title(DIMENSION_DISPLAY.get(metric, metric), fontsize=11)  # type: ignore
        ax.set_ylabel("Vidējais rādītājs", fontsize=9)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

        vmin, vmax = values.min(), values.max()
        spread = vmax - vmin
        if spread < 1e-6:
            pad = max(vmax * 0.05, 0.01)
            ax.set_ylim(vmax - pad, vmax + pad)
        else:
            pad = spread * 0.25
            ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.5)  # type: ignore
        for c in colors
    ]
    fig.legend(
        handles,
        labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        ncol=legend_ncol,
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


def plot_temporal_consistency(means, models_sorted, colors, labels):
    _plot_metric_group(
        means=means,
        models_sorted=models_sorted,
        colors=colors,
        labels=labels,
        metrics=TEMPORAL_DIMENSIONS,
        suptitle="Modeļu salīdzinājums: temporālā konsekvence",
        filename="overall_temporal_consistency.png",
        nrows=2,
        ncols=2,
        figsize=(10, 8),
        legend_loc="upper center",
        legend_bbox=(0.5, -0.02),
        legend_ncol=3,
    )


def plot_visual_quality(means, models_sorted, colors, labels):
    _plot_metric_group(
        means=means,
        models_sorted=models_sorted,
        colors=colors,
        labels=labels,
        metrics=VISUAL_QUALITY_DIMENSIONS,
        suptitle="Modeļu salīdzinājums: vizuālā kvalitāte",
        filename="overall_visual_quality.png",
        nrows=1,
        ncols=2,
        figsize=(10, 4.5),
        legend_loc="upper center",
        legend_bbox=(0.5, -0.05),
        legend_ncol=3,
    )


def plot_semantic_alignment(means, models_sorted, colors, labels):
    _plot_metric_group(
        means=means,
        models_sorted=models_sorted,
        colors=colors,
        labels=labels,
        metrics=SEMANTIC_DIMENSIONS,
        suptitle="Modeļu salīdzinājums: semantiskā atbilstība",
        filename="overall_semantic_alignment.png",
        nrows=1,
        ncols=1,
        figsize=(6, 4.5),
        legend_loc="upper center",
        legend_bbox=(0.5, -0.05),
        legend_ncol=3,
    )


def main():
    setup_plot_style()

    df = build_combined_evaluation_results()

    if df is not None and not df.empty:
        avg_time_df, model_display_order, raw_model_order = prepare_time_taken_data(df)
        plot_time_taken(avg_time_df, model_display_order, raw_model_order)

        means, models_sorted, colors, labels = prepare_overall_evaluation_stats(df)
        plot_temporal_consistency(means, models_sorted, colors, labels)
        plot_visual_quality(means, models_sorted, colors, labels)
        plot_semantic_alignment(means, models_sorted, colors, labels)

        print("Plots generated successfully in the figures directory.")
    else:
        print("Error: No data available to plot.")


if __name__ == "__main__":
    main()
