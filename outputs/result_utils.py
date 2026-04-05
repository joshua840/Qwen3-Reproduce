import re
import socket
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {
    "videomme": 2700,
    "mlvu": 2174,
    "mvbench": None,
    "longvideobench": 1337,
}

SCALING_TP = {8: 875, 16: 1750, 32: 3500, 64: 7000, 128: 14000, 256: 28000, 512: 56000}

# Map directory names to canonical dataset names used in kvpress
DATASET_NAME_MAP = {
    "longvideobench": "longvideobench",
    "mlvu": "mlvu",
    "video-mme": "videomme",
    "videomme": "videomme",
}


def parse_experiment_dir(exp_dir):
    """Parse experiment info from directory structure: {model}/{dataset}/{tp_mf}/"""
    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    # Expected: outputs/{model}/{dataset}/{tp_mf}
    tp_mf_dir = exp_dir.name
    dataset_dir = exp_dir.parent.name
    model_dir = exp_dir.parent.parent.name

    # Parse tp and mf from directory name like "tp1750_mf16"
    m = re.match(r"tp(\d+)_mf(\d+)", tp_mf_dir)
    if not m:
        return None

    tp = int(m.group(1))
    mf = int(m.group(2))

    dataset = DATASET_NAME_MAP.get(dataset_dir.lower(), dataset_dir.lower())

    return {
        "dataset": dataset,
        "model": model_dir,
        "press": "full_kv",
        "compression_ratio": "cr0.00",
        "total_pixels": tp,
        "max_frames": mf,
        "press_kwargs": "",
    }


def make_key(info):
    """Create a unique key for an experiment."""
    return (
        f"{info['press']}|{info['compression_ratio']}|{info['model']}"
        f"|{info['dataset']}|tp{info['total_pixels']}_mf{info['max_frames']}"
        f"|{info['press_kwargs']}"
    )


def build_summary(results_dir):
    """Scan experiments, build summary DataFrame, and save to summary.csv.

    Directory structure: {results_dir}/{model}/{dataset}/{tp_mf}/score.tsv

    Returns:
        summary_df: DataFrame with accuracy results
        incomplete: list of dirnames without score.tsv
    """
    results_dir = Path(results_dir)
    complete = {}
    incomplete = []
    seen_paths = set()

    for p in results_dir.rglob("score.tsv"):
        exp_dir = p.parent
        if exp_dir in seen_paths:
            continue
        seen_paths.add(exp_dir)

        info = parse_experiment_dir(exp_dir)
        if info is None:
            incomplete.append(exp_dir.name)
            continue

        df = pd.read_csv(p, sep="\t")
        correct = int(df[df["score"] > 0]["score"].sum())
        complete[make_key(info)] = {
            "press": info["press"],
            "cr": info["compression_ratio"],
            "model": info["model"],
            "dataset": info["dataset"],
            "mf": info["max_frames"],
            "tp": info["total_pixels"],
            "press_kwargs": info["press_kwargs"],
            "correct": correct,
            "total": len(df),
            "rejected": int((df["score"] == -1).sum()),
            "accuracy": round(correct / len(df) * 100, 2),
        }

    summary_df = pd.DataFrame(list(complete.values()))
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["model", "dataset", "press", "press_kwargs", "mf"]).reset_index(drop=True)

    server_name = socket.gethostname().split(".")[0]
    csv_name = f"summary_fullkv_{server_name}.csv"
    summary_df.to_csv(results_dir / csv_name, index=False)
    print(f"Saved {csv_name} ({len(summary_df)} rows, {len(incomplete)} incomplete)")

    return summary_df, incomplete


if __name__ == "__main__":
    build_summary(".")
