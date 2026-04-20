"""
Bootstrap confidence intervals for reintegration *event-level* means from a saved
`reintegration_detailed_fold*.json` file (written by train.py when markov + dev/test run).

Resamples reintegration events with replacement (same n as observed) and recomputes the
mean of per-event deltas at a chosen offset (binary gap at +0 by default).

Example:
  python -m reintegration.scripts.bootstrap_reintegration_json \\
      --json path/to/reintegration_detailed_fold1.json \\
      --split test --offset 0 --n-bootstrap 10000 --seed 0

Holdout client:
  python -m reintegration.scripts.bootstrap_reintegration_json \\
      --json path/to/reintegration_detailed_fold1.json \\
      --holdout-client 8 --offset 0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _get_delta_list(block: dict, offset: int) -> list[float]:
    d = block.get("delta_by_offset") or {}
    key_str = str(offset)
    if key_str in d:
        raw = d[key_str]
    elif offset in d:
        raw = d[offset]
    else:
        raise KeyError(
            f"offset {offset} not in delta_by_offset keys {list(d.keys())[:20]}..."
        )
    if not isinstance(raw, list):
        raise TypeError(f"delta_by_offset[{offset}] must be a list, got {type(raw)}")
    out = []
    for x in raw:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            continue
        out.append(float(x))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to reintegration_detailed_fold*.json",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split name under ['splits']: dev, test, test_all_zeros_audio",
    )
    p.add_argument(
        "--holdout-client",
        type=str,
        default=None,
        help="If set, use holdout[<id>] instead of splits[--split]",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Recovery lag k for mean of per-event (correct_s - correct_m) deltas",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    p.add_argument(
        "--ci",
        type=float,
        nargs=2,
        default=(2.5, 97.5),
        metavar=("LOW", "HIGH"),
        help="Percentile endpoints for CI (default 2.5 97.5)",
    )
    args = p.parse_args()
    n_boot = args.n_bootstrap

    data = json.loads(args.json.read_text(encoding="utf-8"))
    if args.holdout_client is not None:
        ho = data.get("holdout") or {}
        cid = str(args.holdout_client)
        if cid not in ho:
            raise KeyError(f"holdout client {cid!r} not in file; have {sorted(ho.keys())}")
        block = ho[cid]
    else:
        splits = data.get("splits") or {}
        if args.split not in splits:
            raise KeyError(f"split {args.split!r} not in file; have {list(splits.keys())}")
        block = splits[args.split]

    deltas = _get_delta_list(block, args.offset)
    n = len(deltas)
    if n == 0:
        print("No finite deltas after filtering NaN; aborting.")
        return

    obs_mean = float(np.mean(deltas))
    rng = np.random.default_rng(args.seed)
    arr = np.asarray(deltas, dtype=np.float64)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = float(np.mean(sample))

    lo, hi = np.percentile(boots, list(args.ci))

    meta = data.get("meta") or {}
    print(
        json.dumps(
            {
                "json_file": str(args.json),
                "split": None if args.holdout_client else args.split,
                "holdout_client": args.holdout_client,
                "offset": args.offset,
                "n_events": n,
                "observed_mean_delta": obs_mean,
                "bootstrap_mean_of_means": float(np.mean(boots)),
                "ci_percentiles": list(args.ci),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n_bootstrap": n_boot,
                "seed": args.seed,
                "meta_fold_idx": meta.get("fold_idx"),
                "meta_client_schedule_seed": meta.get("client_schedule_seed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
