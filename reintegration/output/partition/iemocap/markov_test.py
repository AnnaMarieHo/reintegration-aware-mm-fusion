"""
markov_mask_simulation.py
=========================
Simulates and visualises Markov-chain modality missingness masks
over your actual MELD scene length distribution.

Usage:
    python markov_mask_simulation.py --partition path/to/partition.json

Produces 4 plots:
    1. Absence run length distribution per P(a→a)
    2. % scenes containing ≥1 reintegration event per P(a→a)
    3. Mean position of first reintegration event per P(a→a)
    4. Binary availability heatmap for 15 sampled scenes


python markov_test.py \
  --partition partition.json \
  --p_stay_present 0.75 \
  --n_draws 100 \
  --heatmap_p 0.7 
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict


def generate_markov_mask(scene_len: int,
                         p_stay_absent: float,
                         p_stay_present: float,
                         seed: int = None) -> np.ndarray:
    """
    Generate a binary availability mask of length scene_len.

    State:  1 = modality present
            0 = modality absent

    Transition probabilities:
        P(present → present) = p_stay_present
        P(present → absent)  = 1 - p_stay_present
        P(absent  → absent)  = p_stay_absent
        P(absent  → present) = 1 - p_stay_absent   ← reintegration event

    Args:
        scene_len:        number of utterances in the scene
        p_stay_absent:    probability of remaining absent (controls burstiness)
        p_stay_present:   probability of remaining present
        seed:             optional RNG seed for reproducibility

    Returns:
        np.ndarray of shape (scene_len,) with values in {0, 1}
    """
    rng = np.random.default_rng(seed)

    mask = np.ones(scene_len, dtype=int)   # start fully present

    # Sample initial state: bias toward present at t=0
    state = 1 if rng.random() < p_stay_present else 0
    mask[0] = state

    for t in range(1, scene_len):
        if state == 1:   # currently present
            state = 1 if rng.random() < p_stay_present else 0
        else:            # currently absent
            state = 0 if rng.random() < p_stay_absent else 1
        mask[t] = state

    return mask


def find_reintegration_events(mask: np.ndarray) -> list:
    """
    Return list of timestep indices where a reintegration event occurs.
    A reintegration event is defined as:  mask[t-1] == 0  and  mask[t] == 1
    i.e. the modality transitions from absent to present.
    """
    events = []
    for t in range(1, len(mask)):
        if mask[t - 1] == 0 and mask[t] == 1:
            events.append(t)
    return events


def get_absence_run_lengths(mask: np.ndarray) -> list:
    """
    Extract lengths of all contiguous absence runs (consecutive 0s) in mask.
    """
    runs = []
    run = 0
    for v in mask:
        if v == 0:
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    return runs



def run_simulation(scene_lengths: list,
                   p_stay_absent_values: list,
                   p_stay_present: float = 0.85,
                   n_draws: int = 100,
                   seed: int = 42) -> dict:
    """
    For each P(a→a) value, simulate n_draws masks per scene and collect:
        - all absence run lengths
        - whether the scene contained ≥1 reintegration event
        - position of first reintegration event (if any)
        - full masks for a sample of scenes (for heatmap)

    Args:
        scene_lengths:          list of integer scene lengths (from partition)
        p_stay_absent_values:   list of P(a→a) candidates to sweep
        p_stay_present:         P(p→p), fixed across sweep
        n_draws:                number of stochastic mask draws per scene
        seed:                   base RNG seed

    Returns:
        dict keyed by p_stay_absent value, each containing result arrays
    """
    results = {}

    for p_aa in p_stay_absent_values:
        run_lengths       = []        # all absence run lengths observed
        has_reint         = []        # bool per (scene, draw)
        first_reint_pos   = []        # position of first reint (if exists)
        scene_coverage    = []        # fraction of scenes with ≥1 reint event

        rng_base = np.random.default_rng(seed)

        for i, slen in enumerate(scene_lengths):
            scene_has_reint = 0
            for draw in range(n_draws):
                draw_seed = int(rng_base.integers(0, 1e7))
                mask = generate_markov_mask(slen, p_aa, p_stay_present, seed=draw_seed)

                # Absence runs
                run_lengths.extend(get_absence_run_lengths(mask))

                # Reintegration events
                events = find_reintegration_events(mask)
                if events:
                    scene_has_reint += 1
                    first_reint_pos.append(events[0])
                    has_reint.append(1)
                else:
                    has_reint.append(0)

            scene_coverage.append(scene_has_reint / n_draws)

        results[p_aa] = {
            "run_lengths":      run_lengths,
            "has_reint":        has_reint,
            "first_reint_pos":  first_reint_pos,
            "scene_coverage":   scene_coverage,   # per-scene fraction across draws
            "pct_with_reint":   100 * np.mean(has_reint),
            "mean_first_reint": np.mean(first_reint_pos) if first_reint_pos else np.nan,
            "mean_run_length":  np.mean(run_lengths) if run_lengths else 0,
        }

    return results



PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]


def plot_run_length_distributions(results: dict, p_values: list, ax):
    """
    Violin plot of absence run length distributions per P(a→a).
    """
    data = [results[p]["run_lengths"] for p in p_values]
    parts = ax.violinplot(data, positions=range(len(p_values)),
                          showmedians=True, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    parts["cbars"].set_color("#334155")
    parts["cmaxes"].set_color("#334155")
    parts["cmins"].set_color("#334155")

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"P(a→a)={p}" for p in p_values], fontsize=9)
    ax.set_ylabel("Absence Run Length (utterances)")
    ax.set_title("Absence Run Length Distribution per P(a→a)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, None)
    ax.axhline(y=1, color="#475569", linestyle="--", linewidth=0.8, alpha=0.5,
               label="run length = 1")
    ax.legend(fontsize=8)


def plot_reintegration_coverage(results: dict, p_values: list, ax):
    """
    Bar chart: % of (scene, draw) pairs containing ≥1 reintegration event.
    """
    pct_values = [results[p]["pct_with_reint"] for p in p_values]
    bars = ax.bar(range(len(p_values)), pct_values,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(p_values))],
                  alpha=0.85, edgecolor="#1e293b", linewidth=1.2)

    for bar, pct in zip(bars, pct_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9,
                color="white", fontweight="bold")

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"P(a→a)={p}" for p in p_values], fontsize=9)
    ax.set_ylabel("% Sequences with ≥1 Reintegration Event")
    ax.set_title("Reintegration Event Coverage per P(a→a)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=80, color="#f59e0b", linestyle="--", linewidth=1,
               label="80% coverage threshold")
    ax.legend(fontsize=8)


def plot_first_reintegration_position(results: dict, p_values: list,
                                       scene_lengths: list, ax):
    """
    Box plot of first reintegration event position per P(a→a).
    Overlays mean scene length as a reference line.
    """
    data = [results[p]["first_reint_pos"] for p in p_values]
    bp = ax.boxplot(data, positions=range(len(p_values)),
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.75)

    mean_len = np.mean(scene_lengths)
    ax.axhline(y=mean_len, color="#f59e0b", linestyle="--", linewidth=1.2,
               label=f"mean scene length ({mean_len:.1f})")
    ax.axhline(y=min(scene_lengths), color="#ef4444", linestyle=":",
               linewidth=1, label=f"min scene length ({min(scene_lengths)})")

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"P(a→a)={p}" for p in p_values], fontsize=9)
    ax.set_ylabel("Timestep of First Reintegration Event (t)")
    ax.set_title("First Reintegration Position per P(a→a)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)


def plot_availability_heatmap(scene_lengths: list,
                               p_stay_absent: float,
                               p_stay_present: float,
                               n_scenes: int = 15,
                               seed: int = 42,
                               ax=None):
    """
    Binary heatmap showing modality availability masks for n_scenes sampled
    scenes at a chosen P(a→a). Green=present, red=absent, star=reintegration.
    """
    rng = np.random.default_rng(seed)
    sampled_lens = rng.choice(scene_lengths, size=n_scenes, replace=False)
    max_len = max(sampled_lens)

    grid = np.full((n_scenes, max_len), np.nan)
    reint_markers = []

    for i, slen in enumerate(sampled_lens):
        s = int(rng.integers(0, 1e7))
        mask = generate_markov_mask(int(slen), p_stay_absent, p_stay_present, seed=s)
        grid[i, :slen] = mask
        for t in find_reintegration_events(mask):
            reint_markers.append((i, t))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#1e293b")   # NaN = padding = dark background

    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Mark reintegration events
    for (row, col) in reint_markers:
        ax.plot(col, row, marker="*", color="white", markersize=7,
                markeredgecolor="#1e293b", markeredgewidth=0.5)

    ax.set_xlabel("Utterance Position (t)")
    ax.set_ylabel("Scene (sampled)")
    ax.set_title(f"Availability Masks — P(a→a)={p_stay_absent}, "
                 f"P(p→p)={p_stay_present}\n"
                 f"(★ = reintegration event)", fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_scenes))
    ax.set_yticklabels([f"len={l}" for l in sampled_lens], fontsize=7)

    present_patch = mpatches.Patch(color=cmap(1.0), label="present")
    absent_patch  = mpatches.Patch(color=cmap(0.0), label="absent")
    ax.legend(handles=[present_patch, absent_patch], loc="lower right", fontsize=8)


def plot_summary_table(results: dict, p_values: list, ax):
    """
    Renders a clean summary table of key statistics per P(a→a).
    """
    ax.axis("off")
    headers = ["P(a→a)", "Mean run length", "% with reint event",
               "Mean first reint pos"]
    rows = []
    for p in p_values:
        r = results[p]
        rows.append([
            str(p),
            f"{r['mean_run_length']:.2f}",
            f"{r['pct_with_reint']:.1f}%",
            f"{r['mean_first_reint']:.2f}" if not np.isnan(r['mean_first_reint']) else "N/A"
        ])

    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        if row == 0:
            cell.set_facecolor("#6366f1")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1e293b" if row % 2 == 0 else "#0f172a")
            cell.set_text_props(color="#e2e8f0")

    ax.set_title("Summary Statistics", fontsize=11, fontweight="bold",
                 color="white", pad=12)


def main():
    parser = argparse.ArgumentParser(description="Markov missingness mask simulation")
    parser.add_argument("--partition", type=str, required=True,
                        help="Path to partition.json")
    parser.add_argument("--p_stay_present", type=float, default=0.85,
                        help="P(present→present), fixed across sweep (default: 0.85)")
    parser.add_argument("--n_draws", type=int, default=100,
                        help="Mask draws per scene per P(a→a) (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap_p", type=float, default=0.7,
                        help="P(a→a) to use for the heatmap visualisation (default: 0.7)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output plots")
    args = parser.parse_args()

    with open(args.partition) as f:
        partition = json.load(f)

    # Extract scene lengths from test split (primary evaluation target)
    test_scenes  = partition["test"]
    train_scenes = [s for k, v in partition.items()
                    if k not in ("dev", "test")
                    for s in v]

    test_lengths  = [len(scene) for scene in test_scenes]
    train_lengths = [len(scene) for scene in train_scenes]

    print(f"Test  scenes : {len(test_lengths)}  "
          f"(min={min(test_lengths)}, max={max(test_lengths)}, "
          f"mean={np.mean(test_lengths):.1f})")
    print(f"Train scenes : {len(train_lengths)}  "
          f"(min={min(train_lengths)}, max={max(train_lengths)}, "
          f"mean={np.mean(train_lengths):.1f})")

    p_stay_absent_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\nRunning simulation sweep...")
    results = run_simulation(
        scene_lengths       = test_lengths,
        p_stay_absent_values= p_stay_absent_values,
        p_stay_present      = args.p_stay_present,
        n_draws             = args.n_draws,
        seed                = args.seed,
    )

    print("\n" + "="*65)
    print(f"{'P(a→a)':<10} {'Mean run':>12} {'% reint':>12} {'Mean 1st pos':>14}")
    print("-"*65)
    for p in p_stay_absent_values:
        r = results[p]
        print(f"{p:<10} {r['mean_run_length']:>12.2f} "
              f"{r['pct_with_reint']:>11.1f}% "
              f"{r['mean_first_reint']:>14.2f}")
    print("="*65)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    plot_run_length_distributions(results, p_stay_absent_values, axes[0, 0])
    plot_reintegration_coverage(results, p_stay_absent_values, axes[0, 1])
    plot_first_reintegration_position(results, p_stay_absent_values,
                                       test_lengths, axes[1, 0])
    plot_summary_table(results, p_stay_absent_values, axes[1, 1])

    fig.suptitle(
        f"Markov Missingness Sweep  |  P(p→p)={args.p_stay_present}  |  "
        f"{args.n_draws} draws/scene  |  n={len(test_lengths)} test scenes",
        fontsize=13, fontweight="bold", color="#f1f5f9", y=1.01
    )
    plt.tight_layout()
    out1 = Path(args.output_dir) / "markov_sweep.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print(f"\nSaved: {out1}")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#1e293b")
    ax2.tick_params(colors="#94a3b8")
    ax2.xaxis.label.set_color("#94a3b8")
    ax2.yaxis.label.set_color("#94a3b8")
    ax2.title.set_color("#f1f5f9")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")

    plot_availability_heatmap(
        scene_lengths   = test_lengths,
        p_stay_absent   = args.heatmap_p,
        p_stay_present  = args.p_stay_present,
        n_scenes        = 15,
        seed            = args.seed,
        ax              = ax2,
    )
    plt.tight_layout()
    out2 = Path(args.output_dir) / "markov_heatmap.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print(f"Saved: {out2}")
    plt.close()

    print("\nDone. Inspect the summary table to choose your P(a→a) parameter.")
    print("Rule of thumb:")
    print("  - Choose highest P(a→a) where % reint coverage stays above ~85%")
    print("  - Verify mean first reint position < mean scene length")
    print("  - Visual check: heatmap should look bursty, not i.i.d.")


if __name__ == "__main__":
    main()