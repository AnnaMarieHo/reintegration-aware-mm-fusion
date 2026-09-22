#!/usr/bin/env python3
"""
Diagnose per-session scene/utterance distribution across holdout folds.

Usage:
    python diagnose_partition_splits.py
    python diagnose_partition_splits.py --root reintegration/output/partition
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SPLITS = ("train", "dev", "test")
SESSIONS = tuple(f"Ses0{i}" for i in range(1, 6))
SPEAKER_KEY_RE = re.compile(r"^Ses\d{2}[FM]$")


def session_from_wav(wav_name: str) -> str | None:
    # Example: Ses03M_impro02_M010.wav -> Ses03
    m = re.match(r"^(Ses\d{2})[FM]_", wav_name)
    return m.group(1) if m else None


def init_counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        split: {sess: {"scenes": 0, "utterances": 0} for sess in SESSIONS}
        for split in SPLITS
    }


def summarize_partition(partition_json: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    data = json.loads(partition_json.read_text(encoding="utf-8"))
    counts = init_counts()

    # Train is stored by speaker key at top level.
    for key, scenes in data.items():
        if not SPEAKER_KEY_RE.match(key):
            continue
        sess = key[:5]  # Ses01F -> Ses01
        counts["train"][sess]["scenes"] += len(scenes)
        counts["train"][sess]["utterances"] += sum(len(scene) for scene in scenes)

    # Dev/Test are stored as list of scenes; infer session from first utterance filename.
    for split in ("dev", "test"):
        for scene in data.get(split, []):
            if not scene:
                continue
            wav_name = scene[0][0]
            sess = session_from_wav(wav_name)
            if sess is None:
                continue
            counts[split][sess]["scenes"] += 1
            counts[split][sess]["utterances"] += len(scene)

    return counts


def extract_holdout_id(folder_name: str) -> int:
    m = re.search(r"holdout_ses_(\d+)$", folder_name)
    return int(m.group(1)) if m else -1


def find_partition_files(root: Path) -> List[Tuple[int, Path]]:
    results: List[Tuple[int, Path]] = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("holdout_ses_"):
            continue
        holdout_id = extract_holdout_id(d.name)
        pj = d / "partition" / "iemocap" / "partition.json"
        if holdout_id > 0 and pj.exists():
            results.append((holdout_id, pj))
    results.sort(key=lambda x: x[0])
    return results


def print_summary(holdout_id: int, counts: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    print(f"\n=== Holdout Ses0{holdout_id} ===")
    print("split  session  scenes  utterances")
    print("-----  -------  ------  ----------")
    for split in SPLITS:
        for sess in SESSIONS:
            s = counts[split][sess]["scenes"]
            u = counts[split][sess]["utterances"]
            if s == 0 and u == 0:
                continue
            print(f"{split:<5}  {sess:<7}  {s:>6}  {u:>10}")

    # Fold-level totals are useful to compare against partition logs.
    print("\nTotals by split:")
    for split in SPLITS:
        total_scenes = sum(counts[split][sess]["scenes"] for sess in SESSIONS)
        total_utts = sum(counts[split][sess]["utterances"] for sess in SESSIONS)
        print(f"  {split:<5}: {total_scenes:>4} scenes, {total_utts:>5} utterances")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-session train/dev/test scene+utterance counts for each holdout fold."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent,
        help="Path containing holdout_ses_* directories.",
    )
    args = parser.parse_args()

    partition_files = find_partition_files(args.root)
    if not partition_files:
        raise FileNotFoundError(f"No holdout partition.json files found under: {args.root}")

    for holdout_id, partition_json in partition_files:
        counts = summarize_partition(partition_json)
        print_summary(holdout_id, counts)


if __name__ == "__main__":
    main()
