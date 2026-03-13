"""
Prepare IEMOCAP WAVs from local HF Parquet files.

Expected input:
  - One or more Parquet files containing a flat table with at least:
      * 'file'           (e.g. 'Ses01F_impro01_F000.wav')
      * 'audio'          (HF-style dict with raw bytes and path)

We extract raw audio from the Parquet `audio` column and write it to WAV
files named by the original 'file' column, under:

  {output_dir}/raw_iemocap/wavs/{file}

This script does NOT build partition.json; partitioning is handled separately
by `repartition_iemocap_from_parquet.py`.

Usage:

  python -m my_extensions.reintegration.scripts.prepare_iemocap \\
    --parquet_dir /path/to/parquet_folder \\
    --output_dir  /path/to/reintegration/output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError("Please install soundfile to handle IEMOCAP audio: pip install soundfile") from e

# IEMOCAP 6-class mapping (align with constants.num_class_dict['iemocap'] = 6)
IEMOCAP_LABEL_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "excited": 4,
    "frustrated": 5,
}
# Common alternate names in HF / IEMOCAP
IEMOCAP_LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "excitement": "excited",
    "frustration": "frustrated",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare IEMOCAP WAVs from local Parquet files.")
    parser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help="Directory containing one or more IEMOCAP *.parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root (partition + raw_iemocap/wavs). Default: reintegration/output",
    )
    parser.add_argument(
        "--scene_size",
        type=int,
        default=30,
        help="Utterances per scene (default 30).",
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
        help="Fraction of data for dev (default 0.1).",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of data for test (default 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/dev/test split.",
    )
    return parser.parse_args()


def get_label(emotion, row_idx: int) -> int:
    if isinstance(emotion, (int, np.integer)):
        if 0 <= emotion <= 5:
            return int(emotion)
        return -1
    s = str(emotion).strip().lower()
    if s in IEMOCAP_LABEL_MAP:
        return IEMOCAP_LABEL_MAP[s]
    s = IEMOCAP_LABEL_ALIASES.get(s, s)
    return IEMOCAP_LABEL_MAP.get(s, -1)


def build_wavs_from_parquet(
    output_dir: Path,
    parquet_dir: Path,
) -> None:
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    # Load all parquet files into a single DataFrame
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Collect rows with valid filenames and raw audio from the HF-style `audio` column.
    wav_dir = output_dir / "raw_iemocap" / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for i, row in df.iterrows():
        fname = row.get("file")
        if not isinstance(fname, str) or not fname.strip():
            continue
        audio = row.get("audio")
        if audio is None:
            continue

        # HF parquet for audio often stores a dict with raw bytes and a path:
        #   {'bytes': b'....', 'path': 'Ses01F_impro01_F000.wav'}
        # We decode bytes with soundfile to recover waveform + sampling rate.
        arr = None
        sr = 16000
        if isinstance(audio, dict):
            data_bytes = audio.get("bytes", None)
            if data_bytes is not None:
                with BytesIO(data_bytes) as bio:
                    wav_arr, wav_sr = sf.read(bio, dtype="float32")
                if wav_arr.ndim > 1:
                    wav_arr = wav_arr.mean(axis=1)
                arr = wav_arr
                sr = int(wav_sr)
            else:
                # Fallback: older HF-style dict with 'array' and optional 'sampling_rate'
                arr = audio.get("array", None)
                sr = int(audio.get("sampling_rate", 16000))
        else:
            # Fallback: treat as raw array-like with default sr
            arr = np.asarray(audio)

        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue

        out_path = wav_dir / fname
        sf.write(str(out_path), arr, sr)
        n_written += 1

    if n_written == 0:
        raise RuntimeError("No valid IEMOCAP rows (check 'file' and 'audio' columns).")

    print(f"WAVs written under: {wav_dir} (total {n_written})")


def main():
    args = parse_args()
    if args.output_dir is None:
        # Default: reintegration/output relative to this package
        args.output_dir = Path(__file__).resolve().parent.parent.parent / "output"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_wavs_from_parquet(
        output_dir=output_dir,
        parquet_dir=Path(args.parquet_dir),
    )


if __name__ == "__main__":
    main()
