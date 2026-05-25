import json
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "evaluation_results"
CLIP_FILE = EVAL_DIR / "clip_scores.json"
TIMES_FILE = DATA_DIR / "generation_times.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

EXPECTED_MODELS = {
    "hunyuanvideo_1_5_distilled",
    "ltx_2_19b_distilled",
    "wan_2_2_5b",
}

MODEL_DISPLAY = {
    "hunyuanvideo_1_5_distilled": "HunyuanVideo 1.5",
    "ltx_2_19b_distilled": "LTX-Video 2.0",
    "wan_2_2_5b": "Wan 2.2 5B",
}

DIMENSION_DISPLAY = {
    "subject_consistency": "Subject Consistency",
    "background_consistency": "Background Consistency",
    "motion_smoothness": "Motion Smoothness",
    "dynamic_degree": "Dynamic Degree",
    "aesthetic_quality": "Aesthetic Quality",
    "imaging_quality": "Imaging Quality",
    "clip_score": "CLIP Score",
}


TEMPORAL_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
]
VISUAL_QUALITY_DIMENSIONS = [
    "aesthetic_quality",
    "imaging_quality",
]
SEMANTIC_DIMENSIONS = [
    "clip_score",
]
DIMENSION_ORDER = TEMPORAL_DIMENSIONS + VISUAL_QUALITY_DIMENSIONS + SEMANTIC_DIMENSIONS

LENGTH_ORDER = ["5s", "10s", "15s"]
VARIATION_ORDER = ["short", "medium", "long"]

MODEL_COLORS = {
    "hunyuanvideo_1_5_distilled": "#382AD4",
    "ltx_2_19b_distilled": "#10B91E",
    "wan_2_2_5b": "#E2D520",
}

CATEGORY_DISPLAY = {
    "animal": "Dzīvnieki",
    "architecture": "Arhitektūra",
    "food": "Ēdiens",
    "human": "Cilvēki",
    "lifestyle": "Dzīvesveids",
    "plant": "Augi",
    "scenery": "Ainavas",
    "vehicle": "Transports",
}

VARIATION_DISPLAY = {
    "short": "Īsa",
    "medium": "Vidēja",
    "long": "Gara",
}


def parse_video_path(video_path):
    if not isinstance(video_path, str):
        return {"prompt_id": None, "variation": None, "length": None}

    filename = Path(video_path).stem
    match = re.search(r"_(\d+)_(short|medium|long)_(\d+)s$", filename)
    if match:
        return {
            "prompt_id": int(match.group(1)),
            "variation": match.group(2),
            "length": f"{match.group(3)}s",
        }
    return {"prompt_id": None, "variation": None, "length": None}


def load_vbench_results():
    rows = []
    pattern = str(EVAL_DIR / "*" / "*" / "*_eval_results.json")

    for json_path in glob.glob(pattern):
        path = Path(json_path)
        model = path.parent.parent.name
        category = path.parent.name

        if model not in EXPECTED_MODELS:
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        for dimension, value in data.items():
            if isinstance(value, list):
                per_video = value[1]
                for vid in per_video:
                    rows.append(
                        {
                            "model": model,
                            "category": category,
                            "dimension": dimension,
                            "video_path": vid.get("video_path", ""),
                            "score": float(vid.get("video_results", 0)),
                        }
                    )
    return pd.DataFrame(rows)


def load_clip_results():
    with open(CLIP_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        if entry.get("model") not in EXPECTED_MODELS:
            continue
        rows.append(
            {
                "model": entry["model"],
                "category": entry["category"],
                "dimension": "clip_score",
                "video_path": entry["video_path"],
                "score": float(entry["clip_score"]),
            }
        )
    return pd.DataFrame(rows)


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def build_combined_evaluation_results():
    if (TABLES_DIR / "evaluation_results_combined.csv").exists():
        print("Combined evaluation results already exist. Skipping rebuild.")
        return pd.read_csv(TABLES_DIR / "evaluation_results_combined.csv")

    vbench_df = load_vbench_results()
    clip_df = load_clip_results()

    if vbench_df.empty and clip_df.empty:
        print("Error: No evaluation data found.")
        return None

    combined_df = pd.concat([vbench_df, clip_df], ignore_index=True)
    meta_df = pd.DataFrame(combined_df["video_path"].apply(parse_video_path).tolist())
    combined_df = pd.concat([combined_df, meta_df], axis=1).dropna(subset=["prompt_id"])
    combined_df["prompt_id"] = combined_df["prompt_id"].astype(int)

    final_df = combined_df.pivot_table(
        index=["model", "category", "prompt_id", "variation", "length", "video_path"],
        columns="dimension",
        values="score",
        aggfunc="first",
    ).reset_index()
    final_df.columns.name = None

    if TIMES_FILE.exists():
        times_df = pd.read_csv(TIMES_FILE)
        times_df.columns = times_df.columns.str.strip()
        times_df = times_df.rename(
            columns={"version": "variation", "duration": "length"}
        )

        for col in ["model", "category", "variation", "length"]:
            times_df[col] = times_df[col].astype(str)
        times_df["prompt_id"] = times_df["prompt_id"].astype(int)

        final_df = pd.merge(
            final_df,
            times_df[
                [
                    "model",
                    "category",
                    "prompt_id",
                    "variation",
                    "length",
                    "time_seconds",
                ]
            ],
            on=["model", "category", "prompt_id", "variation", "length"],
            how="left",
        )
    else:
        final_df["time_seconds"] = None

    ordered_cols = [
        "model",
        "category",
        "prompt_id",
        "variation",
        "length",
        "time_seconds",
    ]
    metric_cols = [dim for dim in DIMENSION_ORDER if dim in final_df.columns]
    final_df = final_df[ordered_cols + metric_cols + ["video_path"]]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(TABLES_DIR / "evaluation_results_combined.csv", index=False)

    return final_df


def summarize_by_dimension(df, group_col, order=None, output_filename=None, label=None):
    summary = df.groupby(["model", group_col])[DIMENSION_ORDER].mean().round(4)

    if order is not None:
        summary = summary.reindex(order, level=group_col)

    if output_filename:
        summary.to_csv(TABLES_DIR / output_filename)

    label = label or group_col

    return summary


def load_and_prepare_data():
    df = build_combined_evaluation_results()

    if df is None or df.empty:
        raise ValueError("No data to work with")

    df = df[df["model"].isin(EXPECTED_MODELS)].copy()

    if "imaging_quality" in df.columns and df["imaging_quality"].max() > 1.5:
        df["imaging_quality"] = df["imaging_quality"] / 100

    return df
