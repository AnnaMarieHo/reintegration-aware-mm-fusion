"""
Supplementary figures for reintegration results parsed into `parsed_results.txt`.

Reads `holdout_ses_*/parsed_results.txt` and produces:
  1. Heatmap: session × condition, cell = mean ΔUAR_win over folds
  2. Grouped visualization: per-condition bars or points with fold spread
  3. Recovery: mean binary gap at offsets +0..+4 (not macro UAR)

Usage:
  python -m reintegration.scripts.plot_reintegration_supplementary \\
      --partition_root reintegration/output/partition \\
      --out_dir reintegration/output/partition/reint_supp_figs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from reintegration.scripts.parse_parsed_results import (
    DEFAULT_CONDITION_ORDER,
    SHORT_LABELS,
    RunBlock,
    parse_parsed_results_file,
)


def discover_sessions(partition_root: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (session_num, path_to_parsed_results)."""
    out: List[Tuple[int, Path]] = []
    for p in sorted(partition_root.glob("holdout_ses_*")):
        m = re.search(r"holdout_ses_(\d+)", p.name)
        if not m:
            continue
        parsed = p / "parsed_results.txt"
        if parsed.is_file():
            out.append((int(m.group(1)), parsed))
    out.sort(key=lambda x: x[0])
    return out


def load_all_runs(
    sessions: List[Tuple[int, Path]],
    condition_order: Tuple[str, ...],
) -> Dict[str, Dict[int, RunBlock]]:
    """
    condition_key -> session_num -> RunBlock (last block if duplicate keys;
    normally one per file).
    """
    by_c_s: Dict[str, Dict[int, RunBlock]] = {c: {} for c in condition_order}
    for ses_num, path in sessions:
        blocks = parse_parsed_results_file(path)
        for b in blocks:
            if b.condition_key in by_c_s:
                by_c_s[b.condition_key][ses_num] = b
    return by_c_s


def build_heatmap_matrix(
    by_c_s: Dict[str, Dict[int, RunBlock]],
    sessions: List[Tuple[int, Path]],
    condition_order: Tuple[str, ...],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Matrix shape (n_cond, n_ses): mean fold value per cell."""
    ses_nums = [s[0] for s in sessions]
    col_labels = [f"Ses.{n}" for n in ses_nums]
    row_labels = [SHORT_LABELS.get(c, c) for c in condition_order]
    M = np.full((len(condition_order), len(ses_nums)), np.nan, dtype=float)
    for i, c in enumerate(condition_order):
        for j, sn in enumerate(ses_nums):
            blk = by_c_s[c].get(sn)
            if blk is None:
                continue
            M[i, j] = float(np.mean(blk.delta_uar_window))
    return M, row_labels, col_labels


def plot_heatmap(
    M: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    out_path: Path,
    title: str = r"Mean $\Delta\mathrm{UAR}_{win}$ (%)" + "\n(mean over 5 folds per cell)",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    finite = M[np.isfinite(M)]
    if finite.size == 0:
        vmax, vmin = 1.0, -1.0
    else:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if lo < 0 < hi:
            vmax = max(abs(lo), abs(hi))
            vmin = -vmax
        else:
            vmin, vmax = lo, hi
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Held-out test session")
    ax.set_ylabel("Condition")
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                t = "—"
                color = "0.5"
            else:
                t = f"{v:.2f}"
                span = max(abs(vmax - vmin), 1e-6)
                color = "black" if abs(v - vmin) < 0.35 * span else "white"
            ax.text(j, i, t, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta\mathrm{UAR}_{win}$ (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_session_means_with_folds(
    by_c_s: Dict[str, Dict[int, RunBlock]],
    sessions: List[Tuple[int, Path]],
    condition_order: Tuple[str, ...],
    out_path: Path,
) -> None:
    """
    For each condition: grouped bars = one bar per session (height = mean over folds),
    overlay jittered dots for each fold value.
    """
    ses_nums = [s[0] for s in sessions]
    n_c = len(condition_order)
    n_s = len(ses_nums)
    fig, ax = plt.subplots(figsize=(11, 5))

    width = 0.8 / max(n_s, 1)
    x0 = np.arange(n_c, dtype=float)
    cmap = plt.colormaps["tab10"].resampled(max(n_s, 10))
    rng = np.random.default_rng(12345)

    fold_x_all: List[float] = []
    fold_y_all: List[float] = []

    for j, sn in enumerate(ses_nums):
        offsets = x0 + (j - (n_s - 1) / 2) * width
        heights: List[float] = []
        for ci, c in enumerate(condition_order):
            blk = by_c_s[c].get(sn)
            if blk is None:
                heights.append(0.0)
                continue
            vals = np.array(blk.delta_uar_window, dtype=float)
            heights.append(float(np.mean(vals)))
            xj = offsets[ci] + (rng.random(len(vals)) - 0.5) * 0.1
            fold_x_all.extend(xj.tolist())
            fold_y_all.extend(vals.tolist())

        ax.bar(
            offsets,
            heights,
            width=width * 0.92,
            label=f"Ses.{sn}",
            color=cmap(j),
            alpha=0.88,
            edgecolor="0.2",
            linewidth=0.35,
        )

    ax.scatter(
        fold_x_all,
        fold_y_all,
        s=16,
        c="0.12",
        alpha=0.5,
        zorder=3,
        linewidths=0,
    )

    ax.set_xticks(x0)
    ax.set_xticklabels([SHORT_LABELS.get(c, c) for c in condition_order], fontsize=9)
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_ylabel(r"$\Delta\mathrm{UAR}_{win}$ (%)")
    ax.set_title(
        "Per-session means (bars) with fold-level values (dots)\n"
        r"(macro UAR gap on post-return windows)"
    )
    ax.legend(
        title="Bar = mean over folds",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_grand_mean_by_condition(
    by_c_s: Dict[str, Dict[int, RunBlock]],
    sessions: List[Tuple[int, Path]],
    condition_order: Tuple[str, ...],
    out_path: Path,
) -> None:
    """
    For each condition, pool all folds × sessions → mean binary gap at +k ± std.
    """
    offsets = np.arange(5)
    labels = [f"+{k}" for k in range(5)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in condition_order:
        rows: List[np.ndarray] = []
        for _sn, _p in sessions:
            blk = by_c_s[c].get(_sn)
            if blk is None:
                continue
            for row in blk.recovery:
                if len(row) == 5:
                    rows.append(np.array(row, dtype=float))
        if not rows:
            continue
        mat = np.stack(rows, axis=0)
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(5)
        label = SHORT_LABELS.get(c, c)
        ax.plot(offsets, mu, marker="o", label=label, lw=2)
        ax.fill_between(offsets, mu - sd, mu + sd, alpha=0.18)
    ax.axhline(0.0, color="0.45", ls="--", lw=0.9)
    ax.set_xticks(offsets)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Offset after reintegration (utterances)")
    ax.set_ylabel("Mean binary gap (stable correct − masked correct)")
    ax.set_title(
        "Recovery offsets (pooled folds × sessions)\n"
        "Shaded band: ±1 SD across pooled runs"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_lines_per_session(
    by_c_s: Dict[str, Dict[int, RunBlock]],
    sessions: List[Tuple[int, Path]],
    condition_order: Tuple[str, ...],
    out_path: Path,
) -> None:
    """Small multiples: one row per condition, lines = sessions (mean over folds)."""
    ses_nums = [s[0] for s in sessions]
    n_c = len(condition_order)
    fig, axes = plt.subplots(n_c, 1, figsize=(8, 2.2 * n_c), sharex=True, sharey=True)
    if n_c == 1:
        axes = [axes]
    offs = np.arange(5)
    cmap = plt.colormaps["tab10"].resampled(max(len(ses_nums), 10))
    for i, c in enumerate(condition_order):
        ax = axes[i]
        for j, sn in enumerate(ses_nums):
            blk = by_c_s[c].get(sn)
            if blk is None:
                continue
            mat = np.array(blk.recovery, dtype=float)
            if mat.size == 0:
                continue
            mu = mat.mean(axis=0)
            ax.plot(offs, mu, marker="o", color=cmap(j), label=f"Ses.{sn}", lw=1.8)
        ax.axhline(0.0, color="0.5", ls="--", lw=0.7)
        ax.set_ylabel(SHORT_LABELS.get(c, c), fontsize=9)
    axes[-1].set_xticks(offs)
    axes[-1].set_xticklabels([f"+{k}" for k in range(5)])
    axes[-1].set_xlabel("Offset after reintegration")
    fig.suptitle(
        "Mean binary gap per offset (mean over folds within session)",
        y=1.01,
        fontsize=11,
    )
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="upper center", ncol=min(len(ses_nums), 5), bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--partition_root",
        type=Path,
        default=Path("reintegration/output/partition"),
        help="Directory containing holdout_ses_*/parsed_results.txt",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reintegration/output/partition/reint_supp_figs"),
        help="Output directory for PNG/PDF figures",
    )
    args = ap.parse_args()
    root = args.partition_root
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(root)
    if not sessions:
        raise SystemExit(f"No holdout_ses_*/parsed_results.txt under {root}")

    by_c_s = load_all_runs(sessions, DEFAULT_CONDITION_ORDER)

    M, row_labels, col_labels = build_heatmap_matrix(by_c_s, sessions, DEFAULT_CONDITION_ORDER)
    plot_heatmap(M, row_labels, col_labels, out / "delta_uar_win_heatmap.png")
    plot_heatmap(M, row_labels, col_labels, out / "delta_uar_win_heatmap.pdf")

    plot_grouped_session_means_with_folds(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "delta_uar_win_grouped_sessions.png"
    )
    plot_grouped_session_means_with_folds(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "delta_uar_win_grouped_sessions.pdf"
    )

    plot_recovery_grand_mean_by_condition(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "recovery_binary_gap_grand_mean.png"
    )
    plot_recovery_grand_mean_by_condition(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "recovery_binary_gap_grand_mean.pdf"
    )

    plot_recovery_lines_per_session(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "recovery_binary_gap_per_session.png"
    )
    plot_recovery_lines_per_session(
        by_c_s, sessions, DEFAULT_CONDITION_ORDER, out / "recovery_binary_gap_per_session.pdf"
    )

    print(f"Wrote figures to {out.resolve()}")
    print("Sessions:", [s[0] for s in sessions])
    print("Conditions:", list(DEFAULT_CONDITION_ORDER))


if __name__ == "__main__":
    main()
