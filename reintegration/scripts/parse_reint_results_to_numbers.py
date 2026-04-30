#!/usr/bin/env python3
"""
Parse reintegration aggregate logs (e.g. reint_results_all_conditions.txt) into
fold-wise array summaries matching the style of numbers.txt, or JSON / JSONL
for pandas and other tools.

Handles plain lines and timestamped INFO lines:
  2026-04-18 23:39:19 INFO ==> Reintegration test: ...

Example Usage:
 python -m reintegration.scripts.parse_reint_results_to_numbers   reintegration/output/partition/holdout_ses_3/mask_audio_txt_live/slurm-5273429.err   -o reintegration/output/partition/holdout_ses_3/mask_audio_txt_live/parsed_results_ses3.jsonl --format jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _strip_log_prefix(line: str) -> str:
    line = line.rstrip("\n")
    m = re.match(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+INFO\s+==>\s*(.*)$",
        line,
    )
    if m:
        return m.group(1)
    return line


@dataclass
class FoldRecord:
    n_events: int
    uar_stable_global: float
    uar_masked_global: float
    n_win: int
    uar_stable_window: float
    uar_masked_window: float
    delta_uar_window: float
    recovery: List[Tuple[int, float]]  # (+step, value) in log order
    recovery_counts: List[Tuple[int, int]]  # (+step, n) in log order
    logprob_gap: List[Tuple[int, float]]  # (+step, value)
    kl_stable_masked: List[Tuple[int, float]]  # (+step, value)
    argmax_disagreement: List[Tuple[int, float]]  # (+step, value)
    window_uar_by_offset: List[Tuple[int, int, float, float, float]]  # (+step, n, stable, masked, delta)
    mean_delta: float = 0.0  # Optional; defaults to 0.0 if not present in log
    split: str = ""  # e.g., "dev", "test", "test_all_zeros_audio"


# Old format patterns (for backward compatibility)
RE_MAIN = re.compile(
    r"^Reintegration test:\s*mean_delta=(-?\d+(?:\.\d+)?),\s*"
    r"n_events=(\d+),\s*"
    r"UAR_stable=([\d.]+)%,\s*"
    r"UAR_masked=([\d.]+)%\s*$"
)

RE_WINDOW = re.compile(
    r"^Reintegration test \(window UAR\):\s*"
    r"n_win=(\d+),\s*"
    r"UAR_stable=([\d.]+)%,\s*"
    r"UAR_masked=([\d.]+)%,\s*"
    r"delta_uar_window=(-?[\d.]+)%\s*$"
)

RE_CURVE = re.compile(r"^Test recovery curve:\s*(.+)\s*$")
RE_CURVE_PART = re.compile(r"\+(\d+):(-?\d+(?:\.\d+)?)")

# New format patterns with [dev], [test], [test_all_zeros_audio] prefixes
RE_MAIN_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] Reintegration eval:\s*"
    r"n_events=(\d+),\s*"
    r"UAR_stable=([\d.]+)%,\s*"
    r"UAR_masked=([\d.]+)%\s*$"
)

RE_WINDOW_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] Window UAR \(post-reint timestep union\):\s*"
    r"n_win=(\d+),\s*"
    r"UAR_stable=([\d.]+)%,\s*"
    r"UAR_masked=([\d.]+)%,\s*"
    r"delta_uar_window=(-?[\d.]+)%\s*$"
)

RE_CURVE_NEW = re.compile(r"^\[(dev|test|test_all_zeros_audio)\] Recovery curve:\s*(.+)\s*$")
# New format includes (n=X) counts after values: +0:0.1034 (n=145)
RE_CURVE_PART_NEW = re.compile(r"\+(\d+):(-?\d+(?:\.\d+)?|nan)(?:\s*\(n=(\d+)\))?")
RE_LOGPROB_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] Log-prob gap on true class \(nats\):\s*(.+)\s*$"
)
RE_KL_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] KL\(P_stable \|\| P_masked\) \(nats\):\s*(.+)\s*$"
)
RE_ARGMAX_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] Argmax disagreement rate:\s*(.+)\s*$"
)
RE_WINDOW_UAR_BY_OFFSET_NEW = re.compile(
    r"^\[(dev|test|test_all_zeros_audio)\] Window UAR by offset:\s*(.+)\s*$"
)
RE_WINDOW_UAR_OFFSET_PART_NEW = re.compile(
    r"\+(\d+):delta=(-?\d+(?:\.\d+)?|nan)(?:%)?\s*"
    r"\(stable=(-?\d+(?:\.\d+)?|nan)(?:%)?,\s*"
    r"masked=(-?\d+(?:\.\d+)?|nan)(?:%)?,\s*n=(\d+)\)"
)


def _parse_recovery_line(rest: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    for m in RE_CURVE_PART.finditer(rest):
        pairs.append((int(m.group(1)), float(m.group(2))))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_recovery_line_new(rest: str) -> List[Tuple[int, float]]:
    """Parse recovery curve with optional (n=X) counts, e.g., +0:0.1034 (n=145)."""
    pairs: List[Tuple[int, float]] = []
    for m in RE_CURVE_PART_NEW.finditer(rest):
        val_str = m.group(2)
        # Handle 'nan' values gracefully
        if val_str.lower() == "nan":
            continue
        try:
            pairs.append((int(m.group(1)), float(val_str)))
        except ValueError:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_recovery_counts_line_new(rest: str) -> List[Tuple[int, int]]:
    """Parse per-offset sample counts from recovery curve, e.g. +0:0.1034 (n=145)."""
    pairs: List[Tuple[int, int]] = []
    for m in RE_CURVE_PART_NEW.finditer(rest):
        n_str = m.group(3)
        if not n_str:
            continue
        try:
            pairs.append((int(m.group(1)), int(n_str)))
        except ValueError:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_offset_float_line(rest: str) -> List[Tuple[int, float]]:
    """Parse +offset:value lists, skipping nan values."""
    pairs: List[Tuple[int, float]] = []
    for m in RE_CURVE_PART_NEW.finditer(rest):
        val_str = m.group(2)
        if val_str.lower() == "nan":
            continue
        try:
            pairs.append((int(m.group(1)), float(val_str)))
        except ValueError:
            continue
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_window_uar_by_offset_line(rest: str) -> List[Tuple[int, int, float, float, float]]:
    """Parse +offset window-UAR parts into (+k, n, stable, masked, delta)."""
    pairs: List[Tuple[int, int, float, float, float]] = []
    for m in RE_WINDOW_UAR_OFFSET_PART_NEW.finditer(rest):
        step = int(m.group(1))
        delta_str = m.group(2)
        stable_str = m.group(3)
        masked_str = m.group(4)
        n = int(m.group(5))
        delta = float("nan") if delta_str.lower() == "nan" else float(delta_str)
        stable = float("nan") if stable_str.lower() == "nan" else float(stable_str)
        masked = float("nan") if masked_str.lower() == "nan" else float(masked_str)
        pairs.append((step, n, stable, masked, delta))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_folds(lines: Iterable[str]) -> List[FoldRecord]:
    folds: List[FoldRecord] = []
    pending_main: dict | None = None

    for raw in lines:
        line = _strip_log_prefix(raw).strip()
        if not line:
            continue

        # Try old format first
        m_main = RE_MAIN.match(line)
        if m_main:
            pending_main = {
                "mean_delta": float(m_main.group(1)),
                "n_events": int(m_main.group(2)),
                "uar_stable_global": float(m_main.group(3)),
                "uar_masked_global": float(m_main.group(4)),
                "split": "",  # No split in old format
            }
            continue

        # Try new format with [dev]/[test] prefix
        m_main_new = RE_MAIN_NEW.match(line)
        if m_main_new:
            pending_main = {
                "mean_delta": 0.0,  # Will be filled from recovery curve (+0 value)
                "n_events": int(m_main_new.group(2)),
                "uar_stable_global": float(m_main_new.group(3)),
                "uar_masked_global": float(m_main_new.group(4)),
                "split": m_main_new.group(1),
            }
            continue

        m_win = RE_WINDOW.match(line)
        if m_win and pending_main is not None:
            pending_window = {
                "n_win": int(m_win.group(1)),
                "uar_stable_window": float(m_win.group(2)),
                "uar_masked_window": float(m_win.group(3)),
                "delta_uar_window": float(m_win.group(4)),
            }
            # Defer combining until we have recovery curve
            pending_main.update(pending_window)
            continue

        m_win_new = RE_WINDOW_NEW.match(line)
        if m_win_new and pending_main is not None:
            # Validate split matches pending_main
            split = m_win_new.group(1)
            if pending_main.get("split") in ("", split):
                pending_window = {
                    "n_win": int(m_win_new.group(2)),
                    "uar_stable_window": float(m_win_new.group(3)),
                    "uar_masked_window": float(m_win_new.group(4)),
                    "delta_uar_window": float(m_win_new.group(5)),
                    "split": split,
                }
                pending_main.update(pending_window)
            continue

        m_curve = RE_CURVE.match(line)
        if m_curve and pending_main is not None:
            recovery = _parse_recovery_line(m_curve.group(1))
            folds.append(
                FoldRecord(
                    n_events=pending_main["n_events"],
                    mean_delta=pending_main["mean_delta"],
                    uar_stable_global=pending_main["uar_stable_global"],
                    uar_masked_global=pending_main["uar_masked_global"],
                    n_win=pending_main["n_win"],
                    uar_stable_window=pending_main["uar_stable_window"],
                    uar_masked_window=pending_main["uar_masked_window"],
                    delta_uar_window=pending_main["delta_uar_window"],
                    recovery=recovery,
                    recovery_counts=[],
                    logprob_gap=[],
                    kl_stable_masked=[],
                    argmax_disagreement=[],
                    window_uar_by_offset=[],
                    split=pending_main.get("split", ""),
                )
            )
            pending_main = None
            continue

        m_curve_new = RE_CURVE_NEW.match(line)
        if m_curve_new and pending_main is not None:
            split = m_curve_new.group(1)
            if pending_main.get("split") in ("", split):
                recovery = _parse_recovery_line_new(m_curve_new.group(2))
                recovery_counts = _parse_recovery_counts_line_new(m_curve_new.group(2))
                # If mean_delta wasn't set, use the +0 value from recovery curve
                mean_delta = pending_main.get("mean_delta", 0.0)
                if mean_delta == 0.0 and recovery:
                    # Find +0 value
                    for step, val in recovery:
                        if step == 0:
                            mean_delta = val
                            break
                folds.append(
                    FoldRecord(
                        n_events=pending_main["n_events"],
                        mean_delta=mean_delta,
                        uar_stable_global=pending_main["uar_stable_global"],
                        uar_masked_global=pending_main["uar_masked_global"],
                        n_win=pending_main.get("n_win", 0),
                        uar_stable_window=pending_main.get("uar_stable_window", 0.0),
                        uar_masked_window=pending_main.get("uar_masked_window", 0.0),
                        delta_uar_window=pending_main.get("delta_uar_window", 0.0),
                        recovery=recovery,
                        recovery_counts=recovery_counts,
                        logprob_gap=[],
                        kl_stable_masked=[],
                        argmax_disagreement=[],
                        window_uar_by_offset=[],
                        split=split,
                    )
                )
            pending_main = None
            continue

        m_logprob_new = RE_LOGPROB_NEW.match(line)
        if m_logprob_new and folds:
            split = m_logprob_new.group(1)
            if folds[-1].split == split:
                folds[-1].logprob_gap = _parse_offset_float_line(m_logprob_new.group(2))
            continue

        m_kl_new = RE_KL_NEW.match(line)
        if m_kl_new and folds:
            split = m_kl_new.group(1)
            if folds[-1].split == split:
                folds[-1].kl_stable_masked = _parse_offset_float_line(m_kl_new.group(2))
            continue

        m_argmax_new = RE_ARGMAX_NEW.match(line)
        if m_argmax_new and folds:
            split = m_argmax_new.group(1)
            if folds[-1].split == split:
                folds[-1].argmax_disagreement = _parse_offset_float_line(m_argmax_new.group(2))
            continue

        m_window_uar_new = RE_WINDOW_UAR_BY_OFFSET_NEW.match(line)
        if m_window_uar_new and folds:
            split = m_window_uar_new.group(1)
            if folds[-1].split == split:
                folds[-1].window_uar_by_offset = _parse_window_uar_by_offset_line(
                    m_window_uar_new.group(2)
                )
            continue

    return folds


def _fmt_float(x: float, nd: int = 4) -> str:
    s = f"{x:.{nd}f}"
    if s == "-0.0000" or s == "-0.00":
        return "0." + "0" * nd
    return s


def _fmt_arr(xs: Sequence[float], nd: int = 2) -> str:
    inner = ", ".join(_fmt_float(float(v), nd) for v in xs)
    return f"[{inner}]"


def _fmt_recovery_matrix(folds: Sequence[FoldRecord]) -> str:
    rows: List[str] = []
    for i, fr in enumerate(folds):
        parts = [f"+{k}:{_fmt_float(v, 4)}" for k, v in fr.recovery]
        inner = "[" + ", ".join(parts) + "]"
        if i == 0:
            rows.append("[" + inner)  # opens outer [[
        elif i == len(folds) - 1:
            rows.append(inner + "]")  # closes outer ]]
        else:
            rows.append(inner)
    return ",\n".join(rows)


def _fmt_recovery_count_matrix(folds: Sequence[FoldRecord]) -> str:
    rows: List[str] = []
    for i, fr in enumerate(folds):
        parts = [f"+{k}:{n}" for k, n in fr.recovery_counts]
        inner = "[" + ", ".join(parts) + "]"
        if i == 0:
            rows.append("[" + inner)  # opens outer [[
        elif i == len(folds) - 1:
            rows.append(inner + "]")  # closes outer ]]
        else:
            rows.append(inner)
    return ",\n".join(rows)


def _fmt_offset_metric_matrix(
    folds: Sequence[FoldRecord], field_name: str, nd: int = 4
) -> str:
    rows: List[str] = []
    for i, fr in enumerate(folds):
        metric: List[Tuple[int, float]] = getattr(fr, field_name)
        parts = [f"+{k}:{_fmt_float(v, nd)}" for k, v in metric]
        inner = "[" + ", ".join(parts) + "]"
        if i == 0:
            rows.append("[" + inner)  # opens outer [[
        elif i == len(folds) - 1:
            rows.append(inner + "]")  # closes outer ]]
        else:
            rows.append(inner)
    return ",\n".join(rows)


def _fmt_window_uar_by_offset_matrix(folds: Sequence[FoldRecord]) -> str:
    rows: List[str] = []
    for i, fr in enumerate(folds):
        parts = [
            f"+{k}:delta={_fmt_float(delta, 2)}%"
            f"(stable={_fmt_float(stable, 2)}%,masked={_fmt_float(masked, 2)}%,n={n})"
            for k, n, stable, masked, delta in fr.window_uar_by_offset
        ]
        inner = "[" + ", ".join(parts) + "]"
        if i == 0:
            rows.append("[" + inner)  # opens outer [[
        elif i == len(folds) - 1:
            rows.append(inner + "]")  # closes outer ]]
        else:
            rows.append(inner)
    return ",\n".join(rows)


def iter_split_fold_groups(
    folds: Sequence[FoldRecord],
) -> List[Tuple[str, List[FoldRecord]]]:
    """
    Partition folds the same way as format_section: one group per split when
    multiple splits exist; otherwise a single group (split name may be '').
    """
    if not folds:
        return []
    splits = {f.split for f in folds if f.split}
    if len(splits) > 1:
        return [(s, [f for f in folds if f.split == s]) for s in sorted(splits)]
    if len(splits) == 1:
        only = next(iter(splits))
        return [(only, list(folds))]
    return [("", list(folds))]


def fold_record_to_dict(fr: FoldRecord, fold_index: int) -> Dict[str, Any]:
    """JSON-serializable dict for one fold (nested timestep series as list of objects)."""
    out: Dict[str, Any] = {
        "fold_index": fold_index,
        "split": fr.split or None,
        "n_events": fr.n_events,
        "mean_delta": fr.mean_delta,
        "uar_stable_global": fr.uar_stable_global,
        "uar_masked_global": fr.uar_masked_global,
        "n_win": fr.n_win,
        "uar_stable_window": fr.uar_stable_window,
        "uar_masked_window": fr.uar_masked_window,
        "delta_uar_window": fr.delta_uar_window,
        "recovery": [{"timestep": k, "value": v} for k, v in fr.recovery],
    }
    if fr.recovery_counts:
        out["recovery_counts"] = [{"timestep": k, "n": n} for k, n in fr.recovery_counts]
    if fr.logprob_gap:
        out["logprob_gap"] = [{"timestep": k, "value": v} for k, v in fr.logprob_gap]
    if fr.kl_stable_masked:
        out["kl_stable_masked"] = [
            {"timestep": k, "value": v} for k, v in fr.kl_stable_masked
        ]
    if fr.argmax_disagreement:
        out["argmax_disagreement"] = [
            {"timestep": k, "value": v} for k, v in fr.argmax_disagreement
        ]
    if fr.window_uar_by_offset:
        out["window_uar_by_offset"] = [
            {
                "timestep": k,
                "n": n,
                "uar_stable": stable,
                "uar_masked": masked,
                "delta_uar": delta,
            }
            for k, n, stable, masked, delta in fr.window_uar_by_offset
        ]
    return out


def section_to_json_dict(
    job: str, folds: Sequence[FoldRecord], annotations: Sequence[str]
) -> Dict[str, Any]:
    """One slurm section as a dict (used by the aggregate JSON document)."""
    groups: List[Dict[str, Any]] = []
    for split_name, group_folds in iter_split_fold_groups(list(folds)):
        groups.append(
            {
                "split": split_name or None,
                "folds": [
                    fold_record_to_dict(fr, i) for i, fr in enumerate(group_folds)
                ],
            }
        )
    return {
        "job": job,
        "annotations": list(annotations),
        "split_groups": groups,
    }


def build_json_document(
    sections: Sequence[Tuple[str, List[FoldRecord], List[str]]],
    source: str | None = None,
) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "format_version": 1,
        "sections": [
            section_to_json_dict(job, folds, ann)
            for job, folds, ann in sections
        ],
    }
    if source is not None:
        doc["source"] = source
    return doc


def format_jsonl(
    sections: Sequence[Tuple[str, List[FoldRecord], List[str]]],
) -> str:
    """One JSON object per line: one row per fold (job, split, scalars, series)."""
    lines: List[str] = []
    for job, folds, ann in sections:
        ann_list = list(ann)
        for split_name, group_folds in iter_split_fold_groups(list(folds)):
            for i, fr in enumerate(group_folds):
                row = fold_record_to_dict(fr, i)
                row["job"] = job
                row["split"] = split_name or None
                row["annotations"] = ann_list
                lines.append(json.dumps(row, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


def format_fold_group(folds: Sequence[FoldRecord]) -> str:
    """Format a group of folds (single split) into the standard output format."""
    if not folds:
        return "(no fold blocks parsed)\n"

    n_events_set = {f.n_events for f in folds}
    n_win_set = {f.n_win for f in folds}
    if len(n_events_set) != 1:
        n_events_str = " / ".join(sorted(str(x) for x in n_events_set))
    else:
        n_events_str = str(next(iter(n_events_set)))

    lines: List[str] = []
    lines.append(f"n_events = {n_events_str}")
    lines.append("idx of these arrays correspond to folds")
    lines.append(
        "mean_delta (average binary stable-vss-masked gap at reintegration - "
        "recovery curve is built from this) = "
        + _fmt_arr([f.mean_delta for f in folds], nd=4)
    )
    lines.append(
        "UAR_stable (global) = " + _fmt_arr([f.uar_stable_global for f in folds])
    )
    lines.append(
        "UAR_masked (global) = " + _fmt_arr([f.uar_masked_global for f in folds])
    )
    lines.append("")
    lines.append("Window:")
    if len(n_win_set) == 1:
        lines.append(f"n_win = {next(iter(n_win_set))}")
    else:
        lines.append(f"n_win = {' / '.join(str(x) for x in sorted(n_win_set))}")
    lines.append(
        "UAR_stable = " + _fmt_arr([f.uar_stable_window for f in folds])
    )
    lines.append(
        "UAR_masked = " + _fmt_arr([f.uar_masked_window for f in folds])
    )
    lines.append(
        "delta_uar_window = " + _fmt_arr([f.delta_uar_window for f in folds])
    )
    lines.append("")
    lines.append("index of first dimension array corresponds to fold")
    lines.append("index of second dimension arrays correspond to timesteps within window")
    lines.append(_fmt_recovery_matrix(folds))
    if any(f.recovery_counts for f in folds):
        lines.append("")
        lines.append("recovery window counts per timestep (n):")
        lines.append("index of first dimension array corresponds to fold")
        lines.append("index of second dimension arrays correspond to timesteps within window")
        lines.append(_fmt_recovery_count_matrix(folds))
    if any(f.logprob_gap for f in folds):
        lines.append("")
        lines.append("log-prob gap on true class (nats):")
        lines.append("index of first dimension array corresponds to fold")
        lines.append("index of second dimension arrays correspond to timesteps within window")
        lines.append(_fmt_offset_metric_matrix(folds, "logprob_gap", nd=4))
    if any(f.kl_stable_masked for f in folds):
        lines.append("")
        lines.append("KL(P_stable || P_masked) (nats):")
        lines.append("index of first dimension array corresponds to fold")
        lines.append("index of second dimension arrays correspond to timesteps within window")
        lines.append(_fmt_offset_metric_matrix(folds, "kl_stable_masked", nd=4))
    if any(f.argmax_disagreement for f in folds):
        lines.append("")
        lines.append("argmax disagreement rate:")
        lines.append("index of first dimension array corresponds to fold")
        lines.append("index of second dimension arrays correspond to timesteps within window")
        lines.append(_fmt_offset_metric_matrix(folds, "argmax_disagreement", nd=4))
    if any(f.window_uar_by_offset for f in folds):
        lines.append("")
        lines.append("window UAR by offset (%):")
        lines.append("index of first dimension array corresponds to fold")
        lines.append("index of second dimension arrays correspond to timesteps within window")
        lines.append(_fmt_window_uar_by_offset_matrix(folds))
    lines.append("")
    return "\n".join(lines)


def format_section(job: str, folds: Sequence[FoldRecord]) -> str:
    if not folds:
        return f"{job}\n\n\n(no fold blocks parsed)\n"

    groups = iter_split_fold_groups(list(folds))
    if len(groups) > 1:
        lines: List[str] = []
        lines.append(job)
        lines.append("")
        for split_name, split_folds in groups:
            lines.append(f"[{split_name}]")
            lines.append(format_fold_group(split_folds))
        return "\n".join(lines) + "\n"
    _, only_folds = groups[0]
    return job + "\n\n" + format_fold_group(only_folds) + "\n"


def split_slurm_sections(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (job_id_line, body) e.g. ('slurm-5272607.err', '...').
    """
    parts = re.split(r"(?m)^(?=slurm-\d+\.err\s*$)", text)
    sections: List[Tuple[str, str]] = []
    for chunk in parts:
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        first = lines[0].strip()
        if not re.match(r"^slurm-\d+\.err$", first):
            continue
        body = "\n".join(lines[1:])
        sections.append((first, body))
    return sections


def _extract_annotations(body: str) -> List[str]:
    """Trailing annotation lines (inference / training / avg window uar)."""
    tail_lines: List[str] = []
    for line in body.splitlines():
        ls = _strip_log_prefix(line).strip()
        if ls.startswith("avg window uar"):
            tail_lines.append(ls)
        elif ls.startswith("inference:"):
            tail_lines.append(ls)
        elif ls.startswith("(training regime:"):
            tail_lines.append(ls)
    seen = set()
    ordered: List[str] = []
    for t in tail_lines:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def gather_sections(path: Path) -> List[Tuple[str, List[FoldRecord], List[str]]]:
    """Parse input file into (job_line, folds, annotations) per slurm section."""
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_sections = split_slurm_sections(text)
    if not raw_sections:
        job_name = path.name
        if re.match(r"^slurm-\d+\.err$", job_name):
            raw_sections = [(job_name, text)]
        else:
            raw_sections = [(str(path), text)]

    out: List[Tuple[str, List[FoldRecord], List[str]]] = []
    for job, body in raw_sections:
        folds = _parse_folds(body.splitlines())
        out.append((job, folds, _extract_annotations(body)))
    return out


def parse_file(path: Path) -> str:
    out_chunks: List[str] = []
    for job, folds, ordered in gather_sections(path):
        out_chunks.append(format_section(job, folds))
        if ordered:
            out_chunks.append("\n".join(ordered) + "\n")
        out_chunks.append("\n\n\n")
    return "".join(out_chunks).rstrip() + "\n"


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert reint_results_all_conditions-style logs to numbers.txt-style arrays."
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to reint_results_all_conditions.txt (or similar)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write result to this file (default: stdout)",
    )
    p.add_argument(
        "--format",
        choices=("txt", "json", "jsonl"),
        default="txt",
        help="Output format: txt (default), json (one document), or jsonl (one fold per line).",
    )
    p.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indent for --format json (default: 2; use 0 for compact).",
    )
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"Not a file: {args.input}", file=sys.stderr)
        return 1

    if args.format == "txt":
        result = parse_file(args.input)
    else:
        sections = gather_sections(args.input)
        if args.format == "json":
            doc = build_json_document(sections, source=str(args.input))
            if args.json_indent <= 0:
                result = json.dumps(doc, separators=(",", ":")) + "\n"
            else:
                result = json.dumps(doc, indent=args.json_indent) + "\n"
        else:
            result = format_jsonl(sections)

    if args.output is not None:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
