"""
Prepare IEMOCAP WAVs from local HF Parquet files (speaker holdout variant).

Expected input:
  - One or more Parquet files containing a flat table with at least:
      * 'file'   (e.g. 'Ses01F_impro01_F000.wav')
      * 'audio'  (HF-style dict with raw bytes and path)

We extract raw audio from the Parquet `audio` column and write it to WAV
files named by the original 'file' column, under:

  {output_dir}/raw_iemocap/wavs/{file}

All sessions are extracted here (including the holdout session).
Partitioning — including which session is held out — is handled separately
by repartition_holdout.py, which reads only the Parquet metadata (not audio).

Typical workflow:
  1. python -m reintegration.scripts.prepare_holdout_iemocap 
         --parquet_dir /path/to/parquet
         --output_dir  /path/to/reintegration/output

  2. python -m reintegration.scripts.repartition_holdout \\
         --parquet_dir  /path/to/parquet \\
         --wav_root     /path/to/reintegration/output/raw_iemocap/wavs \\
         --output_dir   /path/to/reintegration/output \\
         --holdout_session 5

  3. python -m reintegration.scripts.extract_audio_features  ...
     python -m reintegration.scripts.extract_text_features   ...
     (feature extraction keys on speaker IDs from partition.json)
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "Please install soundfile to handle IEMOCAP audio: pip install soundfile"
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract IEMOCAP WAVs from Parquet files (speaker holdout variant)."
    )
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
        help=(
            "Output root; WAVs are written to {output_dir}/raw_iemocap/wavs/. "
            "Defaults to reintegration/output relative to this package."
        ),
    )
    return parser.parse_args()


def build_wavs_from_parquet(output_dir: Path, parquet_dir: Path) -> None:
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)

    wav_dir = output_dir / "raw_iemocap" / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for _, row in df.iterrows():
        fname = row.get("file")
        if not isinstance(fname, str) or not fname.strip():
            continue
        audio = row.get("audio")
        if audio is None:
            continue

        arr = None
        sr  = 16000

        if isinstance(audio, dict):
            data_bytes = audio.get("bytes")
            if data_bytes is not None:
                with BytesIO(data_bytes) as bio:
                    wav_arr, wav_sr = sf.read(bio, dtype="float32")
                if wav_arr.ndim > 1:
                    wav_arr = wav_arr.mean(axis=1)
                arr = wav_arr
                sr  = int(wav_sr)
            else:
                # Older HF-style dict with 'array' key
                arr = audio.get("array")
                sr  = int(audio.get("sampling_rate", 16000))
        else:
            arr = np.asarray(audio)

        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue

        sf.write(str(wav_dir / fname), arr, sr)
        n_written += 1

    if n_written == 0:
        raise RuntimeError(
            "No valid IEMOCAP rows written (check 'file' and 'audio' columns)."
        )

    print(f"WAVs written to: {wav_dir}  (total: {n_written})")


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent.parent.parent / "output"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_wavs_from_parquet(
        output_dir=output_dir,
        parquet_dir=Path(args.parquet_dir),
    )


if __name__ == "__main__":
    main()
