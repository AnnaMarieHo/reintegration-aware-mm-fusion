"""
Repartition IEMOCAP from Parquet metadata + existing WAV folder.

This script assumes:
  - You already have all IEMOCAP audio files as WAVs under a single root folder,
    e.g.  output/raw_iemocap  (files like Ses01F_impro01_F000.wav, ...).
  - You have one or more Parquet files with per-utterance metadata including:
        * 'file'           (e.g. 'Ses01F_impro01_F000.wav')
        * 'major_emotion'  (string label)
        * 'transcription'  (utterance text)

It does NOT decode audio from Parquet; it only uses metadata and points each
utterance to an existing WAV by joining --wav_root / file.

Output:
  - partition/iemocap/partition.json in MELD scene format:
        partition[client_id][scene_idx][utt_idx] =
            [Filename, Path, Label, Utterance, None]
    where client_ids are '0'..'N-1' for train, plus 'dev' and 'test'.

Usage:

  python -m my_extensions.reintegration.scripts.repartition_iemocap \\
    --parquet_dir  /path/to/iemocap_parquet \\
    --wav_root     /path/to/raw_iemocap \\
    --output_dir   /path/to/reintegration/output \\
    --scene_size   25
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# IEMOCAP_LABEL_MAP = {
#     "angry": 0,
#     "happy": 1,
#     "sad": 2,
#     "neutral": 3,
#     "excited": 4,
#     "frustrated": 5,
# }

IEMOCAP_LABEL_MAP = {
    "angry":      0,
    "frustrated": 0,
    "happy":      1,
    "excited":    1,
    "sad":        2,
    "neutral":    3,
}

IEMOCAP_LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "excitement": "excited",
    "frustration": "frustrated",
}


def get_label(emotion: object) -> int:
    """Map major_emotion string to integer 0..5, or -1 if unknown."""
    if isinstance(emotion, (int, np.integer)):
        return int(emotion) if 0 <= emotion <= 5 else -1
    s = str(emotion).strip().lower()
    if s in IEMOCAP_LABEL_MAP:
        return IEMOCAP_LABEL_MAP[s]
    s = IEMOCAP_LABEL_ALIASES.get(s, s)
    return IEMOCAP_LABEL_MAP.get(s, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repartition IEMOCAP from Parquet metadata and existing WAVs."
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help="Directory containing one or more IEMOCAP *.parquet files.",
    )
    parser.add_argument(
        "--wav_root",
        type=str,
        required=True,
        help="Root directory containing IEMOCAP WAV files (e.g. raw_iemocap).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root (where partition/iemocap/partition.json is written).",
    )
    parser.add_argument(
        "--scene_size",
        type=int,
        default=15,
        help="Utterances per scene (default 15; adjust to target ~9–20, mean ~14–15).",
    )
    parser.add_argument(
        "--train_clients",
        type=int,
        default=5,
        help="Number of FL train clients (default 5).",
    )
    parser.add_argument(
        "--dev_frac",
        type=float,
        default=0.1,
        help="Fraction of utterances for dev (default 0.1).",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of utterances for test (default 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_dir = Path(args.parquet_dir)
    wav_root = Path(args.wav_root)
    output_dir = Path(args.output_dir)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    rows = []
    for i, row in df.iterrows():
        # Filename column in your schema is 'file' (e.g. 'Ses01F_impro01_F000.wav')
        fname = row.get("file")
        if not isinstance(fname, str) or not fname.strip():
            continue

        emotion = row.get("major_emotion")
        label = get_label(emotion)
        if label < 0:
            continue

        text = row.get("transcription") or ""
        if not isinstance(text, str):
            text = str(text)

        wav_path = wav_root / fname
        # We don't enforce existence here; extract_audio_feature will warn if missing.

        rows.append(
            {
                "filename": fname,
                "path": str(wav_path),
                "label": int(label),
                "text": text,
            }
        )

    if not rows:
        raise RuntimeError("No valid IEMOCAP rows found (check parquet columns and labels).")

    # ─────────────────────────────────────────────────────────────────────
    # Group by dyadic interaction and chunk within each interaction only.
    # Interaction id: first two components of filename, e.g.
    #   'Ses01F_impro01_F000.wav' -> 'Ses01F_impro01'
    # This preserves conversation boundaries inside scenes.
    # ─────────────────────────────────────────────────────────────────────
    interactions = []
    by_interaction = {}
    for r in rows:
        parts = r["filename"].split("_")
        if len(parts) < 2:
            inter = parts[0]
        else:
            inter = "_".join(parts[:2])
        if inter not in by_interaction:
            by_interaction[inter] = []
            interactions.append(inter)
        by_interaction[inter].append(r)

    def make_scenes_for_interaction(inter_rows):
        """Chunk rows for a single interaction into scenes of size scene_size."""
        scenes = []
        for start in range(0, len(inter_rows), args.scene_size):
            chunk = inter_rows[start : start + args.scene_size]
            if not chunk:
                continue
            scene = []
            for r in chunk:
                scene.append(
                    [
                        r["filename"],
                        r["path"],
                        r["label"],
                        r["text"],
                        None,
                    ]
                )
            scenes.append(scene)
        return scenes

    # Build a global, ordered scene list where each scene is contained
    # within a single interaction (no cross-boundary scenes).

    all_scenes = []
    for inter in interactions:
        all_scenes.extend(make_scenes_for_interaction(by_interaction[inter]))

    if not all_scenes:
        raise RuntimeError("No scenes could be formed from interactions.")

    # Shuffle scenes at scene level (not utterance level) so dev/test are
    # drawn from all sessions rather than always being the last sessions.
    # Seed is fixed for reproducibility.
    rng = np.random.default_rng(42)
    all_scenes = list(all_scenes)
    all_scenes = [s for s in all_scenes if len(s) >= 3] # remove scenes with less than 3 utterances

    rng.shuffle(all_scenes)

    # Stratified train/dev/test split: sample dev and test scenes uniformly
    # across dominant labels so all splits see the full label distribution.
    def dominant_label(scene):
        labels = [utt[2] for utt in scene]
        return Counter(labels).most_common(1)[0][0]

    # Group scene indices by dominant label
    label_to_indices = {}
    for idx, scene in enumerate(all_scenes):
        lbl = dominant_label(scene)
        label_to_indices.setdefault(lbl, []).append(idx)

    dev_indices, test_indices = set(), set()
    for lbl, indices in label_to_indices.items():
        n_dev  = max(1, int(len(indices) * args.dev_frac))
        n_test = max(1, int(len(indices) * args.test_frac))
        dev_indices.update(indices[:n_dev])
        test_indices.update(indices[n_dev:n_dev + n_test])

    train_scenes = [s for i, s in enumerate(all_scenes)
                    if i not in dev_indices and i not in test_indices]
    dev_scenes   = [all_scenes[i] for i in sorted(dev_indices)]
    test_scenes  = [all_scenes[i] for i in sorted(test_indices)]

    # Stratified client assignment: distribute scenes evenly by dominant label
    # so each client sees a representative label mix. This replaces the old
    # round-robin which gave systematically skewed clients due to interaction ordering.
    label_queues = {}
    for scene in train_scenes:
        lbl = dominant_label(scene)
        label_queues.setdefault(lbl, []).append(scene)

    client_scenes = [[] for _ in range(args.train_clients)]
    client_idx = 0
    for lbl in sorted(label_queues):
        for scene in label_queues[lbl]:
            client_scenes[client_idx % args.train_clients].append(scene)
            client_idx += 1

    partition = {}
    for c in range(args.train_clients):
        partition[str(c)] = client_scenes[c]
    partition["dev"] = dev_scenes
    partition["test"] = test_scenes

    part_dir = output_dir / "partition" / "iemocap"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / "partition.json"

    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(partition, f, indent=2, ensure_ascii=False)

    print(f"Partition written: {part_path}")
    print(
        f"Train scenes: {len(train_scenes)} "
        f"(clients={args.train_clients}), dev scenes: {len(dev_scenes)}, "
        f"test scenes: {len(test_scenes)}"
    )


if __name__ == "__main__":
    main()