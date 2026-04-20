
"""
Repartition IEMOCAP using session-based speaker holdout.

Split strategy (Option B — session holdout):
  - test  : all utterances from the holdout session (default: session 5,
             both speakers Ses05F and Ses05M).
  - dev   : a fraction of the remaining sessions' utterances (sampled at
             the interaction level so conversation boundaries are respected).
  - train : remaining interactions, one FL client per speaker
             (8 speakers from sessions 1-4: Ses01F, Ses01M, ..., Ses04F, Ses04M).

This guarantees zero speaker leakage between train and test: every utterance
in the test split was produced by a speaker who never appears in training.

This script assumes:
  - You already have all IEMOCAP audio files as WAVs under a single root
    folder, e.g. output/raw_iemocap/wavs/
    (files like Ses01F_impro01_F000.wav, ...).
  - You have one or more Parquet files with per-utterance metadata including:
        * 'file'           (e.g. 'Ses01F_impro01_F000.wav')
        * 'major_emotion'  (string label)
        * 'transcription'  (utterance text)

Output:
  - partition/iemocap/partition.json in MELD scene format:
        partition[client_id][scene_idx][utt_idx] =
            [Filename, Path, Label, Utterance, None]
    where client_ids are speaker strings for train clients
    (e.g. 'Ses01F', 'Ses01M', ..., 'Ses04F', 'Ses04M'),
    plus 'dev' and 'test'.

Speaker extraction from filename:
  'Ses01F_impro01_F000.wav'
   ^^^^^^ — session+dyad-gender prefix  (e.g. 'Ses01F')
                   ^ — turn-taker gender suffix ('F' or 'M' from 'F000')
  actual speaker = session prefix (first 5 chars) + turn-taker gender
  e.g. 'Ses01F_impro01_F000.wav' → speaker 'Ses01F'
       'Ses01F_impro01_M001.wav' → speaker 'Ses01M'

Usage:

python -m my_extensions.reintegration.scripts.repartition_holdout   --parquet_dir  /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/IEMOCAP   --wav_root     /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/raw_iemocap/wavs   --output_dir   /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/partition/holdout --holdout_session 5 --scene_size   15 --dev_frac 0.15

python -m my_extensions.reintegration.scripts.repartition_holdout   --parquet_dir  /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/IEMOCAP   --wav_root     /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/raw_iemocap/wavs   --output_dir   /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/partition/holdout_ses_1 --holdout_session 1 --scene_size   15 --dev_frac 0.15

python -m my_extensions.reintegration.scripts.repartition_holdout   --parquet_dir  /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/IEMOCAP   --wav_root     /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/raw_iemocap/wavs   --output_dir   /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/partition/holdout_ses_2 --holdout_session 2 --scene_size   15 --dev_frac 0.15

python -m my_extensions.reintegration.scripts.repartition_holdout   --parquet_dir  /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/IEMOCAP   --wav_root     /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/raw_iemocap/wavs   --output_dir   /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/partition/holdout_ses_3 --holdout_session 3 --scene_size   15 --dev_frac 0.15

python -m my_extensions.reintegration.scripts.repartition_holdout   --parquet_dir  /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/IEMOCAP   --wav_root     /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/raw_iemocap/wavs   --output_dir   /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions/reintegration/output/partition/holdout_ses_4 --holdout_session 4 --scene_size   15 --dev_frac 0.15


  python -m reintegration.scripts.repartition_holdout \\
    --parquet_dir  /path/to/iemocap_parquet \\
    --wav_root     /path/to/raw_iemocap/wavs \\
    --output_dir   /path/to/reintegration/output \\
    --holdout_session 5 \\
    --scene_size   15 \\
    --dev_frac     0.15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IEMOCAP_LABEL_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "excited": 4,
    "frustrated": 5,
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


def get_speaker(filename: str) -> str:
    """
    Extract the actual speaker ID from an IEMOCAP filename.

    Filename format: Ses<SS><G>_<type><NN>_<TG><UUU>.wav
      SS  = session number (01-05)
      G   = dyad-level gender label (F or M) — identifies the dyad, NOT the speaker
      TG  = turn-taker gender (F or M) — identifies who spoke this utterance

    Speaker = first 5 characters of the session prefix + turn-taker gender.

    Examples:
      'Ses01F_impro01_F000.wav' → 'Ses01F'  (session 1, female spoke)
      'Ses01F_impro01_M001.wav' → 'Ses01M'  (session 1, male spoke)
      'Ses03M_script01_M002.wav' → 'Ses03M'
      'Ses03M_script01_F003.wav' → 'Ses03F'
    """
    parts = filename.replace(".wav", "").split("_")
    # print(parts)
    # parts[0] = 'Ses01F', parts[-1] = 'F000' or 'M001'
    session_prefix = parts[0][:5]          # e.g. 'Ses01'
    turn_gender    = parts[-1][0].upper()  # 'F' or 'M'
    return session_prefix + turn_gender    # e.g. 'Ses01F' or 'Ses01M'


def get_session_number(filename: str) -> int:
    """Extract the session number (1-5) from an IEMOCAP filename."""
    # 'Ses01F_impro01_F000.wav' → 1
    return int(filename[3:5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repartition IEMOCAP with session-based speaker holdout. "
            "The holdout session is withheld entirely as the test split; "
            "remaining sessions are split into one FL client per speaker."
        )
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
        help="Root directory containing IEMOCAP WAV files (e.g. raw_iemocap/wavs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root (where partition/iemocap/partition.json is written).",
    )
    parser.add_argument(
        "--holdout_session",
        type=int,
        # default=5,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="IEMOCAP session number to hold out as the test set (default: 5).",
    )
    parser.add_argument(
        "--scene_size",
        type=int,
        default=15,
        help="Utterances per scene (default 15).",
    )
    parser.add_argument(
        "--dev_frac",
        type=float,
        default=0.15,
        help=(
            "Fraction of training-session interactions to reserve for dev "
            "(sampled at the interaction level; default 0.15)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dev interaction sampling (default 42).",
    )
    return parser.parse_args()


def make_scenes(rows: list, scene_size: int) -> list:
    """
    Chunk a list of utterance dicts into scenes of up to scene_size utterances.
    Each scene is a list of [Filename, Path, Label, Utterance, None] entries.
    """
    scenes = []
    for start in range(0, len(rows), scene_size):
        chunk = rows[start : start + scene_size]
        if not chunk:
            continue
        scenes.append(
            [[r["filename"], r["path"], r["label"], r["text"], None] for r in chunk]
        )
    return scenes


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    parquet_dir = Path(args.parquet_dir)
    wav_root    = Path(args.wav_root)
    output_dir  = Path(args.output_dir)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df  = pd.concat(dfs, ignore_index=True)

    # ── 1. Parse all valid rows ───────────────────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        fname = row.get("file")
        if not isinstance(fname, str) or not fname.strip():
            continue
        label = get_label(row.get("major_emotion"))
        if label < 0:
            continue
        text = row.get("transcription") or ""
        if not isinstance(text, str):
            text = str(text)
        rows.append(
            {
                "filename": fname,
                "path":     str(wav_root / fname),
                "label":    int(label),
                "text":     text,
                "session":  get_session_number(fname),
                "speaker":  get_speaker(fname),
            }
        )

    if not rows:
        raise RuntimeError(
            "No valid IEMOCAP rows found (check parquet columns and labels)."
        )

    # ── 2. Split rows by session ──────────────────────────────────────────────
    holdout_rows = [r for r in rows if r["session"] == args.holdout_session]
    train_rows   = [r for r in rows if r["session"] != args.holdout_session]

    if not holdout_rows:
        raise RuntimeError(
            f"No utterances found for holdout session {args.holdout_session}."
        )
    if not train_rows:
        raise RuntimeError("No training utterances remain after holdout split.")

    # ── 3. Build interaction groups (preserves conversation boundaries) ───────
    #   Interaction key = first two '_'-delimited parts of the filename stem
    #   e.g. 'Ses01F_impro01_F000.wav' → 'Ses01F_impro01'
    def interaction_key(filename: str) -> str:
        parts = filename.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    # Group training rows by interaction, preserving insertion order
    train_interactions: dict[str, list] = {}
    for r in train_rows:
        key = interaction_key(r["filename"])
        train_interactions.setdefault(key, []).append(r)

    # ── 4. Dev split: sample whole interactions from the training pool ────────
    interaction_keys = list(train_interactions.keys())
    n_dev_interactions = max(1, int(len(interaction_keys) * args.dev_frac))
    dev_interaction_set = set(
        rng.choice(interaction_keys, size=n_dev_interactions, replace=False).tolist()
    )

    dev_rows   = [r for k in interaction_keys if k in dev_interaction_set
                    for r in train_interactions[k]]
    train_rows_final = [r for k in interaction_keys if k not in dev_interaction_set
                          for r in train_interactions[k]]

    # ── 5. Group final training rows by speaker ───────────────────────────────
    #   Each speaker becomes one FL client.
    #   IEMOCAP sessions 1-4 → 8 speakers: Ses01F..Ses04M
    speakers: dict[str, list] = {}
    for r in train_rows_final:
        speakers.setdefault(r["speaker"], []).append(r)

    speaker_ids = sorted(speakers.keys())  # deterministic: Ses01F, Ses01M, ..., Ses04M

    # ── 6. Build scenes within each speaker's interactions ────────────────────
    #   Within each speaker we still respect interaction boundaries so that
    #   scenes never cross conversation turns.
    def scenes_for_speaker(spk_rows: list) -> list:
        by_inter: dict[str, list] = {}
        for r in spk_rows:
            by_inter.setdefault(interaction_key(r["filename"]), []).append(r)
        all_scenes = []
        for inter_rows in by_inter.values():
            all_scenes.extend(make_scenes(inter_rows, args.scene_size))
        return all_scenes

    # Dev and test scenes: group by interaction across all speakers
    def scenes_for_rows(row_list: list) -> list:
        by_inter: dict[str, list] = {}
        for r in row_list:
            by_inter.setdefault(interaction_key(r["filename"]), []).append(r)
        all_scenes = []
        for inter_rows in by_inter.values():
            all_scenes.extend(make_scenes(inter_rows, args.scene_size))
        return all_scenes

    # ── 7. Assemble partition ─────────────────────────────────────────────────
    partition: dict = {}
    for spk in speaker_ids:
        spk_scenes = scenes_for_speaker(speakers[spk])
        if not spk_scenes:
            print(f"  Warning: speaker {spk} produced 0 scenes, skipping.")
            continue
        partition[spk] = spk_scenes

    partition["dev"]  = scenes_for_rows(dev_rows)
    partition["test"] = scenes_for_rows(holdout_rows)

    if not partition["test"]:
        raise RuntimeError(
            f"Holdout session {args.holdout_session} produced no scenes."
        )

    # ── 8. Write partition.json ───────────────────────────────────────────────
    part_dir  = output_dir / "partition" / "iemocap"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / "partition.json"

    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(partition, f, indent=2, ensure_ascii=False)

    # ── 9. Summary ───────────────────────────────────────────────────────────
    n_train_scenes = sum(len(v) for k, v in partition.items() if k not in ("dev", "test"))
    n_dev_scenes   = len(partition["dev"])
    n_test_scenes  = len(partition["test"])
    n_train_utts   = sum(len(s) for k, v in partition.items()
                         if k not in ("dev", "test") for s in v)
    n_dev_utts     = sum(len(s) for s in partition["dev"])
    n_test_utts    = sum(len(s) for s in partition["test"])

    print(f"\nPartition written: {part_path}")
    print(f"  Holdout session : {args.holdout_session}  (speakers: "
          f"{', '.join(sorted({r['speaker'] for r in holdout_rows}))})")
    print(f"  Train clients   : {len(speaker_ids)} speakers — {', '.join(speaker_ids)}")
    print(f"  Train scenes    : {n_train_scenes}  ({n_train_utts} utterances)")
    print(f"  Dev scenes      : {n_dev_scenes}  ({n_dev_utts} utterances, "
          f"{n_dev_interactions} interactions)")
    print(f"  Test scenes     : {n_test_scenes}  ({n_test_utts} utterances)")
    print()
    for spk in speaker_ids:
        if spk in partition:
            sc = len(partition[spk])
            ut = sum(len(s) for s in partition[spk])
            print(f"    {spk}: {sc} scenes, {ut} utterances")


if __name__ == "__main__":
    main()
