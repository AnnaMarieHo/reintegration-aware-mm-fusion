#!/usr/bin/env python3
"""
Parse reintegration aggregate logs (e.g. reint_results_all_conditions.txt) into
fold-wise array summaries matching the style of numbers.txt.

Handles plain lines and timestamped INFO lines:
  2026-04-18 23:39:19 INFO ==> Reintegration test: ...
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


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
    mean_delta: float
    uar_stable_global: float
    uar_masked_global: float
    n_win: int
    uar_stable_window: float
    uar_masked_window: float
    delta_uar_window: float
    recovery: List[Tuple[int, float]]  # (+step, value) in log order


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


def _parse_recovery_line(rest: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    for m in RE_CURVE_PART.finditer(rest):
        pairs.append((int(m.group(1)), float(m.group(2))))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _parse_folds(lines: Iterable[str]) -> List[FoldRecord]:
    folds: List[FoldRecord] = []
    pending_main: dict | None = None

    for raw in lines:
        line = _strip_log_prefix(raw).strip()
        if not line:
            continue

        m_main = RE_MAIN.match(line)
        if m_main:
            pending_main = {
                "mean_delta": float(m_main.group(1)),
                "n_events": int(m_main.group(2)),
                "uar_stable_global": float(m_main.group(3)),
                "uar_masked_global": float(m_main.group(4)),
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
                )
            )
            pending_main = None
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


def format_section(job: str, folds: Sequence[FoldRecord]) -> str:
    if not folds:
        return f"{job}\n\n\n(no fold blocks parsed)\n"

    n_events_set = {f.n_events for f in folds}
    n_win_set = {f.n_win for f in folds}
    if len(n_events_set) != 1:
        n_events_str = " / ".join(sorted(str(x) for x in n_events_set))
    else:
        n_events_str = str(next(iter(n_events_set)))

    lines: List[str] = []
    lines.append(job)
    lines.append("")
    lines.append("")
    lines.append(f"n_events = {n_events_str}")
    lines.append("idx of these arrays correspond to folds")
    lines.append(
        "mean_delta (average binary stable-vss-masked gap at reintegration - "
        "recvery curve is built from this) = "
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
    lines.append("")
    return "\n".join(lines) + "\n"


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


def parse_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    out_chunks: List[str] = []

    for job, body in split_slurm_sections(text):
        folds = _parse_folds(body.splitlines())
        out_chunks.append(format_section(job, folds))

        # Pass through trailing annotation lines after the last fold block
        tail_lines: List[str] = []
        for line in body.splitlines():
            ls = _strip_log_prefix(line).strip()
            if ls.startswith("avg window uar"):
                tail_lines.append(ls)
            elif ls.startswith("inference:"):
                tail_lines.append(ls)
            elif ls.startswith("(training regime:"):
                tail_lines.append(ls)

        if tail_lines:
            # De-dup while preserving order of unique blocks
            seen = set()
            ordered: List[str] = []
            for t in tail_lines:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
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
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"Not a file: {args.input}", file=sys.stderr)
        return 1

    result = parse_file(args.input)
    if args.output is not None:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
