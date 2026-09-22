"""
Plot partition statistics from a scene-formatted partition.json.

Expected structure (same as repartition scripts):
    partition[client_or_split_id][scene_idx][utt_idx] =
        [filename, path, emotion_id, text, ...]

Metrics per key (each FL client plus 'dev' and 'test' when present):
  - utterances: total rows across all scenes
  - scenes:     number of scene lists
  - transitions: within each scene, count of adjacent utterance pairs whose
                 emotion ids differ (emotion change events along the timeline)
  - emotions:   histogram of emotion ids over utterances

Usage (single file — all keys):
  python -m reintegration.scripts.visualize_partition_stats \\
      --partition path/to/partition.json \\
      --out_dir   path/to/figures

Compare **test** split only across several partition files (e.g. holdout folds):
  python -m reintegration.scripts.visualize_partition_stats \\
      --partitions path/to/fold_a/.../partition.json path/to/fold_b/.../partition.json \\
      --out_dir path/to/figures
  Optional: --labels holdout_ses_1 holdout_ses_2  (defaults inferred from paths)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


IEMOCAP_ID2NAME = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "excited",
    5: "frustrated",
}


def ordered_partition_keys(keys: List[str]) -> List[str]:
    """Stable order: sorted non-special keys, then dev, then test."""
    special = {"dev", "test"}
    clients = sorted(k for k in keys if k not in special)
    tail = [k for k in ("dev", "test") if k in keys]
    return clients + tail


def count_scene_stats(scenes: List[List[Any]]) -> Tuple[int, int, int, Counter]:
    n_scenes = len(scenes)
    n_utts = 0
    n_transitions = 0
    emo = Counter()

    for scene in scenes:
        n_utts += len(scene)
        for row in scene:
            if len(row) >= 3:
                lab = row[2]
                if lab is not None:
                    emo[int(lab)] += 1
        for i in range(len(scene) - 1):
            a = scene[i][2] if len(scene[i]) > 2 else None
            b = scene[i + 1][2] if len(scene[i + 1]) > 2 else None
            if a is None or b is None:
                continue
            if int(a) != int(b):
                n_transitions += 1

    return n_scenes, n_utts, n_transitions, emo


def default_compare_label(part_path: Path) -> str:
    """Prefer folder above .../partition/<dataset>/partition.json (e.g. holdout_ses_1)."""
    p = part_path.resolve()
    if p.name == "partition.json" and p.parent.parent.name == "partition":
        return p.parent.parent.parent.name
    return p.stem


def stats_for_split(partition: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    if split not in partition:
        return None
    scenes = partition[split]
    n_scenes, n_utts, n_trans, emo = count_scene_stats(scenes)
    return {
        "scenes": n_scenes,
        "utterances": n_utts,
        "transitions": n_trans,
        "emotions": emo,
    }


def gather_stats(partition: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in ordered_partition_keys(list(partition.keys())):
        scenes = partition[key]
        n_scenes, n_utts, n_trans, emo = count_scene_stats(scenes)
        out[key] = {
            "scenes": n_scenes,
            "utterances": n_utts,
            "transitions": n_trans,
            "emotions": emo,
        }
    return out


def plot_bars(
    keys: List[str],
    series: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.45), 4.5))
    names = list(series.keys())
    for i, name in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * width, series[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_emotion_stacked(
    keys: List[str],
    stats: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str = "Emotion distribution (stacked fractions)",
) -> None:
    """Stacked bar: fraction of utterances per emotion, per partition key."""
    all_ids = sorted(
        {e for s in stats.values() for e in s["emotions"].keys()},
        key=int,
    )
    if not all_ids:
        return
    labels = [IEMOCAP_ID2NAME.get(int(i), str(i)) for i in all_ids]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.45), 5))
    bottom = np.zeros(len(keys))
    for j, eid in enumerate(all_ids):
        counts = np.array([stats[k]["emotions"].get(int(eid), 0) for k in keys], dtype=float)
        totals = np.array([stats[k]["utterances"] for k in keys], dtype=float)
        fracs = np.divide(counts, np.maximum(totals, 1.0))
        ax.bar(x, fracs, bottom=bottom, label=labels[j], color=f"C{j % 10}")
        bottom += fracs
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("fraction of utterances")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emotion_heatmap(
    keys: List[str],
    stats: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str = "Emotion counts (utterances per class)",
) -> None:
    """Rows = partition keys, cols = emotion id — raw utterance counts."""
    all_ids = sorted({e for s in stats.values() for e in s["emotions"].keys()}, key=int)
    if not all_ids:
        return
    mat = np.array(
        [[stats[k]["emotions"].get(int(eid), 0) for eid in all_ids] for k in keys],
        dtype=float,
    )
    col_labels = [IEMOCAP_ID2NAME.get(int(i), str(i)) for i in all_ids]
    fig, ax = plt.subplots(figsize=(max(6, len(all_ids) * 1.0), max(4, len(keys) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(all_ids)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_title(title)
    for i in range(len(keys)):
        for j in range(len(all_ids)):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", color="0.2", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_compare_test_splits(
    part_paths: List[Path],
    labels: List[str],
    out_dir: Path,
) -> None:
    """Plots for the 'test' split only, one row per partition file (e.g. per holdout fold)."""
    stats: Dict[str, Dict[str, Any]] = {}
    for label, path in zip(labels, part_paths):
        with open(path, encoding="utf-8") as f:
            partition = json.load(f)
        test_stats = stats_for_split(partition, "test")
        if test_stats is None:
            raise SystemExit(f"No 'test' split in {path}")
        stats[label] = test_stats

    keys = list(stats.keys())
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel in (
        ("utterances", "count"),
        ("scenes", "count"),
        ("transitions", "count"),
    ):
        plot_bars(
            keys,
            {metric: [float(stats[k][metric]) for k in keys]},
            title=f"Test: {metric} per partition",
            ylabel=ylabel,
            out_path=out_dir / f"test_{metric}_per_partition.png",
        )

    plot_emotion_stacked(
        keys,
        stats,
        out_dir / "test_emotions_stacked_fractions.png",
        title="Test: emotion distribution (stacked fractions)",
    )
    plot_emotion_heatmap(
        keys,
        stats,
        out_dir / "test_emotions_count_heatmap.png",
        title="Test: emotion counts (utterances per class)",
    )

    print("Compare: test split across partition files")
    print(f"Figures: {out_dir}")
    print("-" * 72)
    hdr = f"{'label':<20} {'source':<52} {'scenes':>8} {'utts':>8} {'trans':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, path in zip(labels, part_paths):
        s = stats[label]
        src = str(path)
        if len(src) > 50:
            src = "..." + src[-49:]
        print(
            f"{label:<20} {src:<52} {s['scenes']:>8} "
            f"{s['utterances']:>8} {s['transitions']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize partition.json scene/utt/emo stats.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--partition",
        type=Path,
        help="Single partition.json (all keys: clients, dev, test)",
    )
    g.add_argument(
        "--partitions",
        nargs="+",
        type=Path,
        help="Several partition.json files: visualize the **test** split only, one column per file",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for --partitions (same length); default: inferred from each path",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (single-file default: <partition_dir>/partition_stats_figs; "
        "compare default: ./test_partition_compare_figs)",
    )
    args = parser.parse_args()

    if args.partitions is not None:
        part_paths = [p.expanduser().resolve() for p in args.partitions]
        if args.labels is not None:
            if len(args.labels) != len(part_paths):
                raise SystemExit("--labels must have the same length as --partitions")
            labels = list(args.labels)
        else:
            labels = [default_compare_label(p) for p in part_paths]

        out_dir = args.out_dir
        if out_dir is None:
            out_dir = Path("test_partition_compare_figs").resolve()
        else:
            out_dir = Path(out_dir).expanduser().resolve()

        run_compare_test_splits(part_paths, labels, out_dir)
        return

    part_path = args.partition.expanduser().resolve()
    with open(part_path, encoding="utf-8") as f:
        partition = json.load(f)

    stats = gather_stats(partition)
    keys = list(stats.keys())

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = part_path.parent / "partition_stats_figs"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-series bar charts (counts)
    for metric, ylabel in (
        ("utterances", "count"),
        ("scenes", "count"),
        ("transitions", "count"),
    ):
        plot_bars(
            keys,
            {metric: [float(stats[k][metric]) for k in keys]},
            title=f"{metric.capitalize()} per partition key",
            ylabel=ylabel,
            out_path=out_dir / f"{metric}_per_key.png",
        )

    plot_emotion_stacked(keys, stats, out_dir / "emotions_stacked_fractions.png")
    plot_emotion_heatmap(keys, stats, out_dir / "emotions_count_heatmap.png")

    # Console summary
    print(f"Partition: {part_path}")
    print(f"Figures:   {out_dir}")
    print("-" * 72)
    hdr = f"{'key':<12} {'scenes':>8} {'utts':>8} {'trans':>8}"
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        s = stats[k]
        print(f"{k:<12} {s['scenes']:>8} {s['utterances']:>8} {s['transitions']:>8}")


if __name__ == "__main__":
    main()
