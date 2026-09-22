#!/usr/bin/env python3
"""
Plot speaker utterance distributions from partition_holdout_stats.txt.

Usage:
    python plot_partition_holdout_stats.py
    python plot_partition_holdout_stats.py --input partition_holdout_stats.txt --output speaker_distribution.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


@dataclass
class FoldStats:
    holdout_session: int
    holdout_speakers: Sequence[str]
    test_utterances: int
    train_speakers_order: Sequence[str]
    train_utterances: Dict[str, int]


def short_label(full_speaker_id: str) -> str:
    # Ses01F -> S1F
    m = re.match(r"Ses0?(\d)([FM])$", full_speaker_id.strip())
    if not m:
        return full_speaker_id.strip()
    return f"S{m.group(1)}{m.group(2)}"


def parse_partition_stats(text: str) -> List[FoldStats]:
    blocks = [
        b.strip()
        for b in re.split(r"\n\s*\n\s*\n+", text.strip())
        if "Holdout session" in b
    ]

    folds: List[FoldStats] = []
    for block in blocks:
        session_match = re.search(
            r"Holdout session\s*:\s*(\d+)\s*\(speakers:\s*([^)]+)\)",
            block,
        )
        train_clients_match = re.search(r"Train clients\s*:\s*\d+\s*speakers\s*[—-]\s*(.+)", block)
        test_match = re.search(r"Test scenes\s*:\s*\d+\s*\((\d+)\s*utterances\)", block)
        if not session_match or not test_match or not train_clients_match:
            continue

        session_id = int(session_match.group(1))
        holdout_speakers = [s.strip() for s in session_match.group(2).split(",")]
        test_utterances = int(test_match.group(1))
        train_speakers_order = [s.strip() for s in train_clients_match.group(1).split(",")]

        train_counts: Dict[str, int] = {}
        for line in block.splitlines():
            m = re.match(r"\s*(Ses\d{2}[FM]):\s*\d+\s*scenes,\s*(\d+)\s*utterances", line)
            if m:
                train_counts[m.group(1)] = int(m.group(2))

        folds.append(
            FoldStats(
                holdout_session=session_id,
                holdout_speakers=holdout_speakers,
                test_utterances=test_utterances,
                train_speakers_order=train_speakers_order,
                train_utterances=train_counts,
            )
        )

    # Preserve fold ordering exactly as written in the txt file.
    return folds


def build_plot_data(fold: FoldStats):
    # Fixed global speaker index order across all folds:
    # Ses01F, Ses01M, Ses02F, Ses02M, ... , Ses05F, Ses05M
    all_speakers = [f"Ses0{s}{g}" for s in range(1, 6) for g in ("F", "M")]
    holdout_set = set(fold.holdout_speakers)

    # The txt reports total holdout test utterances for the pair, not per speaker.
    # Split equally so the two holdout bars stay at their true speaker indices.
    holdout_each = fold.test_utterances / max(len(fold.holdout_speakers), 1)

    labels = [short_label(s) for s in all_speakers]
    values: List[float] = []
    types: List[str] = []
    for s in all_speakers:
        if s in holdout_set:
            values.append(holdout_each)
            types.append("holdout")
        else:
            values.append(float(fold.train_utterances.get(s, 0)))
            types.append("train")

    return labels, values, types


def make_figure(folds: Sequence[FoldStats], output_path: Path) -> None:
    if not folds:
        raise ValueError("No fold blocks were parsed from the input txt file.")

    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    fold_colors = plt.cm.viridis(np.linspace(0.15, 0.8, n))
    holdout_color = "#d9d9d9"
    holdout_edge = "#bfbfbf"

    for idx, (ax, fold) in enumerate(zip(axes, folds)):
        labels, values, types = build_plot_data(fold)
        x = np.arange(len(labels))
        bar_colors = [holdout_color if t == "holdout" else fold_colors[idx] for t in types]
        bars = ax.bar(x, values, color=bar_colors, width=0.72, alpha=0.75, edgecolor="none")

        for bar, t in zip(bars, types):
            if t == "holdout":
                bar.set_hatch("////")
                bar.set_edgecolor(holdout_edge)
                bar.set_linewidth(0.6)
                bar.set_alpha(0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Holdout\nSes0{fold.holdout_session}", fontsize=11, pad=2)
        ax.grid(axis="y", linestyle="-", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.2)
        ax.spines["bottom"].set_alpha(0.2)

    axes[0].set_ylabel("Utterances", fontsize=11)
    fig.suptitle(
        "Speaker utterance distribution across IEMOCAP cross-validation folds",
        y=0.98,
        fontsize=13,
    )

    legend_elements = [
        Patch(facecolor=fold_colors[min(2, n - 1)], edgecolor="none", alpha=0.75, label="Train speaker"),
        Patch(
            facecolor=holdout_color,
            edgecolor=holdout_edge,
            hatch="////",
            alpha=0.85,
            label="Holdout (test only, split evenly)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.89])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot holdout speaker distribution from txt stats.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("partition_holdout_stats.txt"),
        help="Path to partition_holdout_stats.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("speaker_utterance_distribution.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    folds = parse_partition_stats(text)
    make_figure(folds, args.output)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
