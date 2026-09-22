"""
Aggregate saved outputs from `bootstrap_reintegration_json.py` into a CSV table
and an errorbar plot (mean ± bootstrap percentile CI vs offset).

Each input file must contain one JSON object with at least:
  json_file, offset, observed_mean_delta, ci_low, ci_high, n_events

Save bootstrap stdout to a file, e.g.:
  python -m reintegration.scripts.bootstrap_reintegration_json \\
      --json reintegration/output/.../reintegration_detailed_fold3.json \\
      --split test --seed 2 --offset 0 > boot_fold3_off0.json

Then:
  python -m reintegration.scripts.summarize_bootstrap_results \\
      --out-csv bootstrap_summary.csv --out-plot bootstrap_summary.png \\
      boot_fold3_off0.json boot_fold3_off1.json

Or one JSON Lines file (one JSON object per line):
  python -m reintegration.scripts.summarize_bootstrap_results \\
      --from-jsonl all_bootstraps.jsonl --out-csv summary.csv --out-plot summary.png

If bootstrap JSONs were written under a summary directory, pass full paths or a glob
(quote the glob in the shell):
  python -m reintegration.scripts.summarize_bootstrap_results \\
      --glob 'reintegration/output/.../fold1_summary/boot_fold1_off*.json' \\
      --out-csv .../fold1.csv --out-plot .../fold1.png
"""

from __future__ import annotations

import argparse
import csv
import glob as glob_module
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_KEYS = (
    "json_file",
    "offset",
    "observed_mean_delta",
    "ci_low",
    "ci_high",
    "n_events",
)


def _parse_one_record(obj: Dict[str, Any], source: str) -> Dict[str, Any]:
    missing = [k for k in EXPECTED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"{source}: missing keys {missing}")
    row = {k: obj[k] for k in obj}
    row["_source_path"] = source
    jf = str(row.get("json_file", ""))
    m = re.search(r"fold(\d+)\.json", jf, re.I)
    row["_fold_from_path"] = int(m.group(1)) if m else None
    fold_meta = row.get("meta_fold_idx")
    row["_plot_label"] = (
        f"fold {fold_meta}"
        if fold_meta is not None
        else (f"fold {row['_fold_from_path']}" if row["_fold_from_path"] else Path(jf).name)
    )
    return row


def load_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_records_from_jsonl(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield f"{path}:{i}", json.loads(line)


def expand_json_paths(
    json_files: Sequence[Path],
    glob_patterns: Sequence[str],
) -> List[Path]:
    """Resolve positional paths and --glob patterns into a de-duplicated list."""
    out: List[Path] = []
    for p in json_files:
        out.append(p)
    for pattern in glob_patterns:
        matched = sorted(glob_module.glob(pattern, recursive=True))
        if not matched:
            raise FileNotFoundError(
                f"--glob {pattern!r} matched no files (cwd={Path.cwd()})"
            )
        out.extend(Path(s) for s in matched)
    seen: set[Path] = set()
    deduped: List[Path] = []
    for p in out:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def gather_rows(
    json_files: Sequence[Path],
    jsonl: Optional[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in json_files:
        rows.append(_parse_one_record(load_json_file(p), str(p)))
    if jsonl is not None:
        for src, obj in iter_records_from_jsonl(jsonl):
            rows.append(_parse_one_record(obj, src))
    return rows


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("json_file", "")),
            int(r["offset"]),
            str(r.get("split") or ""),
            str(r.get("holdout_client") or ""),
        ),
    )


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    columns = [
        "json_file",
        "split",
        "holdout_client",
        "offset",
        "n_events",
        "observed_mean_delta",
        "ci_low",
        "ci_high",
        "ci_width",
        "bootstrap_mean_of_means",
        "ci_percentiles",
        "n_bootstrap",
        "seed",
        "meta_fold_idx",
        "meta_client_schedule_seed",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k) for k in columns}
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            flat["ci_width"] = hi - lo
            if flat.get("ci_percentiles") is not None and not isinstance(
                flat["ci_percentiles"], str
            ):
                flat["ci_percentiles"] = json.dumps(flat["ci_percentiles"])
            w.writerow(flat)


def plot_rows(rows: List[Dict[str, Any]], out: Path, title: Optional[str]) -> None:
    by_series: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = str(r["json_file"])
        by_series[key].append(r)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    cmap = plt.get_cmap("tab10")

    for i, (series_key, items) in enumerate(sorted(by_series.items(), key=lambda x: x[0])):
        items = sorted(items, key=lambda r: int(r["offset"]))
        x = np.array([int(r["offset"]) for r in items], dtype=float)
        y = np.array([float(r["observed_mean_delta"]) for r in items])
        lo = np.array([float(r["ci_low"]) for r in items])
        hi = np.array([float(r["ci_high"]) for r in items])
        yerr = np.vstack([y - lo, hi - y])
        label = items[0].get("_plot_label") or Path(series_key).name
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            capsize=4,
            color=cmap(i % 10),
            label=str(label),
            linewidth=1.2,
            markersize=6,
        )

    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_xlabel("Recovery lag (offset)")
    ax.set_ylabel("Mean per-event Δ (bootstrap CI)")
    ax.set_title(title or "Bootstrap mean Δ with percentile CI")
    if len(by_series) > 1:
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(rows: List[Dict[str, Any]], max_rows: int) -> None:
    cols = [
        "offset",
        "n_events",
        "observed_mean_delta",
        "ci_low",
        "ci_high",
        "meta_fold_idx",
    ]
    disp = rows[:max_rows]
    widths = [max(len(c), max(len(f"{r.get(c, '')}") for r in disp)) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in disp:
        print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help="Paths to single-record JSON files (bootstrap script stdout)",
    )
    p.add_argument(
        "--glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob for bootstrap JSON files (repeatable); merged with positional paths",
    )
    p.add_argument(
        "--from-jsonl",
        type=Path,
        default=None,
        help="JSON Lines: one bootstrap result object per line",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    p.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Output figure path (.png or .pdf). Default: same stem as --out-csv with .png",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write CSV",
    )
    p.add_argument(
        "--print-table",
        action="store_true",
        help="Print a compact table to stdout",
    )
    p.add_argument(
        "--print-table-rows",
        type=int,
        default=50,
        help="Max rows to print with --print-table",
    )
    args = p.parse_args()

    if not args.json_files and not args.from_jsonl and not args.glob:
        p.error("Provide JSON file paths, --glob, and/or --from-jsonl")

    try:
        json_paths = expand_json_paths(args.json_files, args.glob)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    missing = [p for p in json_paths if not p.is_file()]
    if missing:
        cwd = Path.cwd().resolve()
        lines = "\n".join(f"  {p}" for p in missing)
        raise SystemExit(
            "Bootstrap JSON file(s) not found (paths are relative to the process cwd):\n"
            f"cwd: {cwd}\n{lines}\n"
            "Use the directory where your shell script wrote them, e.g.\n"
            "  reintegration/output/partition/holdout_ses_1/mask_txt_audio_live/fold1_summary/boot_fold1_off0.json\n"
            "or: --glob '.../fold1_summary/boot_fold1_off*.json'"
        )

    rows = sort_rows(gather_rows(json_paths, args.from_jsonl))
    if not rows:
        raise SystemExit("No records loaded.")

    write_csv(rows, args.out_csv)

    plot_path = args.out_plot
    if plot_path is None and not args.no_plot:
        plot_path = args.out_csv.with_suffix(".png")

    if not args.no_plot and plot_path is not None:
        plot_rows(rows, plot_path, args.title)

    if args.print_table:
        print_table(rows, args.print_table_rows)

    print(
        json.dumps(
            {
                "n_records": len(rows),
                "out_csv": str(args.out_csv),
                "out_plot": None if args.no_plot else str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
