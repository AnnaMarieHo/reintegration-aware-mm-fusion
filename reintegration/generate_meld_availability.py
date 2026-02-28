"""
Precompute temporal availability sidecars for MELD.

Writes one .pkl per (dataset, p_off_on, p_on_off, seed) under
  {output_dir}/simulation_feature/meld/
  availability_markov_{p01}_{p10}_seed{seed}.pkl

Sidecar structure:
  {
    client_id: {
      idx: {
        "mask_a": np.ndarray(T_a, bool),
        "mask_b": np.ndarray(T_b, bool),
        "events_a": np.ndarray(T_a, bool),
        "events_b": np.ndarray(T_b, bool),
        "r_a": np.ndarray(T_a, int32),
        "r_b": np.ndarray(T_b, int32),
      },
      ...
    },
    ...
  }

Usage:
  python -m my_extensions.reintegration.generate_meld_availability \\
    --data_dir /path/to/output \\
    --output_dir /path/to/output \\
    --availability_process markov \\
    --availability_seed 42 \\
    --p_off_on 0.05 --p_on_off 0.05
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from fed_multimodal.dataloader.dataload_manager import DataloadManager

from .availability import (
    TwoStateMarkovParams,
    generate_availability_schedule,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sequence_length(feature_entry) -> int:
    """Infer time dimension from a feature array (last element of data_dict[idx])."""
    if feature_entry is None:
        return 0
    arr = np.asarray(feature_entry)
    if arr.ndim == 2:
        return arr.shape[0]
    if arr.ndim == 3:
        return arr.shape[1]
    return 0


def _make_args(
    data_dir: str,
    dataset: str = "meld",
    audio_feat: str = "mfcc",
    text_feat: str = "mobilebert",
):
    """Minimal args for DataloadManager to resolve MELD feature paths."""
    class Args:
        pass

    args = Args()
    args.dataset = dataset
    args.data_dir = data_dir
    args.audio_feat = audio_feat
    args.text_feat = text_feat
    return args


def generate_meld_availability(
    data_dir: str,
    output_dir: str,
    availability_process: str,
    availability_seed: int,
    p_off_on: float,
    p_on_off: float,
    dataset: str = "meld",
) -> Path:
    """
    Precompute per-sample temporal availability and write sidecar.

    Args:
        data_dir: root containing feature/audio/..., feature/text/... (same as fed_multimodal data_dir).
        output_dir: root for writing simulation_feature/meld/availability_markov_*.pkl.
        availability_process: "markov" (generates sidecar); "bernoulli" skips (use existing sim JSON).
        availability_seed: RNG seed for Markov chains.
        p_off_on: P(OFF -> ON).
        p_on_off: P(ON -> OFF).
        dataset: dataset name, must be "meld".

    Returns:
        Path to the written sidecar .pkl (or None if skipped).
    """
    if availability_process != "markov":
        logger.info(
            "availability_process=%s: not generating temporal sidecar (use existing simulation JSON for bernoulli).",
            availability_process,
        )
        return None

    if dataset != "meld":
        raise ValueError(f"Only dataset='meld' is supported; got {dataset!r}")

    args = _make_args(data_dir=data_dir, dataset=dataset)
    dm = DataloadManager(args)
    dm.get_text_feat_path()
    dm.get_audio_feat_path()
    dm.get_client_ids()

    markov_params = TwoStateMarkovParams(p_off_on=p_off_on, p_on_off=p_on_off)
    sidecar = {}

    for client_id in dm.client_ids:
        audio_dict = dm.load_audio_feat(client_id=client_id)
        text_dict = dm.load_text_feat(client_id=client_id)
        # Both dicts are keyed by the same integer indices.
        indices = sorted(k for k in audio_dict if k in text_dict)
        sidecar[client_id] = {}

        for idx in indices:
            # data_dict[idx] = [..., label, feature_array]; feature at -1
            fe_a = audio_dict[idx][-1]
            fe_b = text_dict[idx][-1]
            T_a = _sequence_length(fe_a)
            T_b = _sequence_length(fe_b)

            # Deterministic seed per (client, sample)
            try:
                seed = hash((client_id, idx)) % (2**32)
            except TypeError:
                seed = hash(str(client_id) + str(idx)) % (2**32)
            seed = (availability_seed + seed) % (2**32)

            a_mask, b_mask, events_a, events_b, r_a, r_b = generate_availability_schedule(
                dataset_name="meld",
                len_a=T_a,
                len_b=T_b,
                markov_params=markov_params,
                seed=seed,
            )
            sidecar[client_id][idx] = {
                "mask_a": a_mask,
                "mask_b": b_mask,
                "events_a": events_a,
                "events_b": events_b,
                "r_a": r_a,
                "r_b": r_b,
            }

    # Output path: output_dir/simulation_feature/meld/availability_markov_{p01}_{p10}_seed{seed}.pkl
    p01_str = str(p_off_on).replace(".", "")
    p10_str = str(p_on_off).replace(".", "")
    out_dir = Path(output_dir).joinpath("simulation_feature", dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(f"availability_markov_{p01_str}_{p10_str}_seed{availability_seed}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Wrote sidecar: %s", out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute MELD temporal availability sidecar (Markov).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing feature/audio/..., feature/text/... (fed_multimodal data_dir).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root for writing simulation_feature/meld/*.pkl; defaults to data_dir.",
    )
    parser.add_argument(
        "--availability_process",
        type=str,
        choices=("bernoulli", "markov"),
        default="markov",
        help="Process type: only 'markov' generates a temporal sidecar.",
    )
    parser.add_argument(
        "--availability_seed",
        type=int,
        default=0,
        help="RNG seed for Markov availability chains.",
    )
    parser.add_argument(
        "--p_off_on",
        type=float,
        default=0.05,
        help="P(OFF -> ON) for 2-state Markov.",
    )
    parser.add_argument(
        "--p_on_off",
        type=float,
        default=0.05,
        help="P(ON -> OFF) for 2-state Markov.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="meld",
        help="Dataset name (only meld supported).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    generate_meld_availability(
        data_dir=args.data_dir,
        output_dir=output_dir,
        availability_process=args.availability_process,
        availability_seed=args.availability_seed,
        p_off_on=args.p_off_on,
        p_on_off=args.p_on_off,
        dataset=args.dataset,
    )
