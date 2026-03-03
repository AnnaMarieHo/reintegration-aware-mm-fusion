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

Experiment design (reint vs no_reint):
    For a clean sample-level comparison, use separate sidecars and runs:
    - Run A: default sidecar -> event-level acc by r_t bucket (aux head curve).
    - Run B: _reint sidecar (--sidecar_variant reint) -> utterance acc on forced-reint samples.
    - Run C: _stable sidecar (--sidecar_variant stable) -> utterance acc on stable control.
    Do not expect a large no_reint group from a single default run; use _stable for control.

Usage for Linux/WSL:
    python -m my_extensions.reintegration.generate_meld_availability \
        --data_dir fed-multimodal/fed_multimodal/output/ \
        --output_dir fed-multimodal/fed_multimodal/output/ \
        --availability_process markov \
        --availability_seed 42 \
        --target_reint_events 2.0
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from fed_multimodal.dataloader.dataload_manager import DataloadManager

from .availability import (
    DEFAULT_TARGET_REINT_EVENTS,
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
    *,
    require_reintegration: bool = False,
    require_no_reintegration: bool = False,
    max_flips: Optional[int] = None,
    target_reint_events: Optional[float] = None,
) -> Optional[Path]:
    """
    Precompute per-sample temporal availability and write sidecar.

    Args:
        data_dir: root containing feature/audio/..., feature/text/... (same as fed_multimodal data_dir).
        output_dir: root for writing simulation_feature/meld/availability_markov_*.pkl.
        availability_process: "markov" (generates sidecar); "bernoulli" skips (use existing sim JSON).
        availability_seed: RNG seed for Markov chains.
        p_off_on: P(OFF -> ON) (used when target_reint_events is None).
        p_on_off: P(ON -> OFF) (used when target_reint_events is None).
        dataset: dataset name, must be "meld".
        target_reint_events: if set, use length-scaled p_off_on per modality so text and audio
            both get ~this many OFF->ON events per sequence (recommended: 2.0).

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
    print(f"Text feat path: {dm.text_feat_path}")
    dm.get_audio_feat_path()
    print(f"Audio feat path: {dm.audio_feat_path}")
    dm.get_client_ids()
    print(f"Client ids: {dm.client_ids}")
    markov_params = TwoStateMarkovParams(p_off_on=p_off_on, p_on_off=p_on_off)
    print(f"Markov params: {markov_params}")
    sidecar = {}
    fallback_count = {"audio": 0, "text": 0}

    for client_id in dm.client_ids:
        audio_dict = dm.load_audio_feat(client_id=client_id)
        text_dict = dm.load_text_feat(client_id=client_id)
        print(f"Audio dict: {len(audio_dict)}")
        print(f"Text dict: {len(text_dict)}")
        # Both dicts are keyed by the same integer indices.

        if isinstance(audio_dict, dict) and isinstance(text_dict, dict):
            # If keys are strings like "0", "1", cast to int for sorting and indexing
            common = set(audio_dict.keys()) & set(text_dict.keys())
            # normalize keys to ints
            indices = sorted([int(k) for k in common])
            key_fn = int  # when indexing, cast back if needed
        else:
            # list case: use positional indices (0..min_len-1)
            n = min(len(audio_dict), len(text_dict))
            indices = list(range(n))
            key_fn = lambda x: x  # identity




        print(f"Indices: {indices}")
        print(f"Indices length: {len(indices)}")
        sidecar[client_id] = {}

        for idx in indices:
            # data_dict[idx] = [..., label, feature_array]; feature at -1
            idx = int(idx)
            fe_a = audio_dict[idx][-1]
            fe_b = text_dict[idx][-1]
            T_a = _sequence_length(fe_a)
            T_b = _sequence_length(fe_b)
            print(f"T_a: {T_a}")
            print(f"T_b: {T_b}")
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
                require_reintegration=require_reintegration,
                require_no_reintegration=require_no_reintegration,
                max_flips=max_flips,
                fallback_count=fallback_count,
                target_reint_events=target_reint_events,
            )
            sidecar[client_id][idx] = {
                "mask_a": a_mask,
                "mask_b": b_mask,
                "events_a": events_a,
                "events_b": events_b,
                "r_a": r_a,
                "r_b": r_b,
            }

    # Output path: .../availability_markov_{p01}_{p10}_seed{seed}[_reint|_stable].pkl
    p01_str = str(p_off_on).replace(".", "")
    p10_str = str(p_on_off).replace(".", "")
    out_dir = Path(output_dir).joinpath("simulation_feature", dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if require_reintegration:
        suffix = "_reint"
    elif require_no_reintegration:
        suffix = "_stable"
    scaling_str = f"_tre{str(target_reint_events).replace('.', '')}" if target_reint_events is not None else ""
    out_path = out_dir.joinpath(
        f"availability_markov_{p01_str}_{p10_str}_seed{availability_seed}{scaling_str}{suffix}.pkl"
    )

    with open(out_path, "wb") as f:
        pickle.dump(sidecar, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Wrote sidecar: %s", out_path)
    if fallback_count["audio"] > 0 or fallback_count["text"] > 0:
        logger.warning(
            "Markov guard fallbacks (non-conforming samples returned): audio=%d, text=%d. "
            "Check whether your stable/reint sidecar is actually conforming.",
            fallback_count["audio"],
            fallback_count["text"],
        )
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
    parser.add_argument(
        "--sidecar_variant",
        type=str,
        choices=("default", "reint", "stable", "both"),
        default="default",
        help="default: no constraint. reint: require >=1 OFF->ON per sample. stable: no flips (control). both: write reint and stable sidecars.",
    )
    parser.add_argument(
        "--target_reint_events",
        type=float,
        default=DEFAULT_TARGET_REINT_EVENTS,
        help="Target expected OFF->ON events per sequence; length-scaled p_off_on per modality (default 2.0). Set to 0 to use raw p_off_on/p_on_off for both.",
    )
    return parser.parse_args()


def _reint_kwargs(variant: str) -> dict:
    if variant == "reint":
        return {"require_reintegration": True, "require_no_reintegration": False, "max_flips": None}
    if variant == "stable":
        return {"require_no_reintegration": True, "max_flips": 0}
    return {"require_reintegration": False, "require_no_reintegration": False, "max_flips": None}


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    target_reint = args.target_reint_events if args.target_reint_events > 0 else None
    if args.sidecar_variant == "both":
        for v in ("reint", "stable"):
            generate_meld_availability(
                data_dir=args.data_dir,
                output_dir=output_dir,
                availability_process=args.availability_process,
                availability_seed=args.availability_seed,
                p_off_on=args.p_off_on,
                p_on_off=args.p_on_off,
                dataset=args.dataset,
                target_reint_events=target_reint,
                **_reint_kwargs(v),
            )
    else:
        generate_meld_availability(
            data_dir=args.data_dir,
            output_dir=output_dir,
            availability_process=args.availability_process,
            availability_seed=args.availability_seed,
            p_off_on=args.p_off_on,
            p_on_off=args.p_on_off,
            dataset=args.dataset,
            target_reint_events=target_reint,
            **_reint_kwargs(args.sidecar_variant),
        )
