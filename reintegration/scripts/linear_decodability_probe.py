"""
Linear decodability probe on scene-GRU hidden states during absence periods.

Trains a multinomial logistic regression on hidden vectors (masked and/or stable)
from reintegration JSON absence_timestep_detail rows, with scene-level GroupKFold
to avoid leakage across timesteps from the same conversation.

Example:
  python -m reintegration.scripts.linear_decodability_probe \\
      --json reintegration/output/.../reintegration_detailed_fold4.json \\
      --split test \\
      --vector masked \\
      --out-csv probe_masked.csv \\
      --print-table
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

CHANCE_UAR = {
    6: 100.0 / 6,
    7: 100.0 / 7,
}


def make_logistic_regression(random_state: int) -> LogisticRegression:
    """Multinomial LR compatible with sklearn <1.8 (multi_class) and >=1.8 (removed)."""
    kwargs: dict[str, Any] = {
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": random_state,
    }
    try:
        LogisticRegression(multi_class="multinomial", **kwargs)
        kwargs["multi_class"] = "multinomial"
    except TypeError:
        pass
    return LogisticRegression(**kwargs)


def load_absence_rows(json_path: Path, split: str) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    if "splits" in payload:
        if split not in payload["splits"]:
            raise KeyError(f"split {split!r} not in JSON; have {list(payload['splits'])}")
        block = payload["splits"][split]
    else:
        block = payload

    rows = block.get("absence_timestep_detail")
    if not rows:
        raise ValueError(
            f"No absence_timestep_detail in {json_path} split={split!r}. "
            "Re-run eval with --reint_save_timestep_detail."
        )
    return rows


def vector_key(name: str) -> str:
    mapping = {
        "masked": "hidden_masked",
        "stable": "hidden_stable",
        "ghost": "hidden_ghost",
    }
    if name not in mapping:
        raise ValueError(f"vector must be one of {list(mapping)}; got {name!r}")
    return mapping[name]


def decode_by_offset(
    rows: list[dict[str, Any]],
    *,
    vector: str,
    n_splits: int,
    max_offset: int | None,
    train_max_offset: int | None,
    test_min_offset: int | None,
    random_state: int,
) -> dict[int, dict[str, float | int]]:
    key = vector_key(vector)
    missing = sum(1 for r in rows if key not in r)
    if missing:
        raise ValueError(
            f"{missing} rows missing {key!r}. "
            "Re-run with --reint_save_timestep_detail (and --reint_ghost_pass for ghost)."
        )

    offsets = sorted({int(r["offset_k"]) for r in rows})
    if max_offset is not None:
        offsets = [k for k in offsets if k <= max_offset]

    results: dict[int, dict[str, float | int]] = {}
    for k in offsets:
        if train_max_offset is not None and test_min_offset is not None:
            train_rows = [r for r in rows if int(r["offset_k"]) <= train_max_offset]
            test_rows = [r for r in rows if int(r["offset_k"]) == k and k >= test_min_offset]
            if k < test_min_offset or not test_rows:
                continue
            split_mode = "holdout_offset"
        else:
            train_rows = [r for r in rows if int(r["offset_k"]) == k]
            test_rows = train_rows
            split_mode = "group_kfold"

        X = np.array([r[key] for r in train_rows], dtype=np.float64)
        y = np.array([int(r["y_true"]) for r in train_rows], dtype=np.int64)
        groups = np.array([int(r["scene_batch_idx"]) for r in train_rows], dtype=np.int64)

        if split_mode == "holdout_offset":
            X_test = np.array([r[key] for r in test_rows], dtype=np.float64)
            y_test = np.array([int(r["y_true"]) for r in test_rows], dtype=np.int64)
            X_train, y_train = X, y
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", make_logistic_regression(random_state)),
            ])
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred) * 100
            uar = recall_score(y_test, pred, average="macro", zero_division=0) * 100
            results[k] = {
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_scenes_test": len({int(r["scene_batch_idx"]) for r in test_rows}),
                "accuracy": float(acc),
                "uar": float(uar),
                "split_mode": split_mode,
            }
            continue

        n_unique_groups = len(np.unique(groups))
        n_folds = min(n_splits, n_unique_groups)
        if n_folds < 2 or len(y) < n_folds:
            results[k] = {
                "n": len(y),
                "n_scenes": n_unique_groups,
                "accuracy": float("nan"),
                "uar": float("nan"),
                "split_mode": split_mode,
            }
            continue

        accs: list[float] = []
        uars: list[float] = []
        gkf = GroupKFold(n_splits=n_folds)
        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", make_logistic_regression(random_state)),
            ])
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y[test_idx], pred) * 100)
            uars.append(
                recall_score(y[test_idx], pred, average="macro", zero_division=0) * 100
            )

        results[k] = {
            "n": len(y),
            "n_scenes": n_unique_groups,
            "n_folds": n_folds,
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "uar": float(np.mean(uars)),
            "uar_std": float(np.std(uars)),
            "split_mode": split_mode,
        }

    return results


def print_table(
    results: dict[int, dict[str, float | int]],
    *,
    num_classes: int,
    vector: str,
) -> None:
    chance = CHANCE_UAR.get(num_classes, float("nan"))
    logging.info("Linear probe on %s hidden states (chance UAR ≈ %.1f%%)", vector, chance)
    for k in sorted(results):
        r = results[k]
        if r.get("split_mode") == "holdout_offset":
            logging.info(
                "  offset %2d: n_train=%s n_test=%s UAR=%.1f%% acc=%.1f%% [%s]",
                k,
                r["n_train"],
                r["n_test"],
                r["uar"],
                r["accuracy"],
                r["split_mode"],
            )
        else:
            logging.info(
                "  offset %2d: n=%s scenes=%s folds=%s UAR=%.1f±%.1f%% acc=%.1f±%.1f%%",
                k,
                r.get("n", "?"),
                r.get("n_scenes", "?"),
                r.get("n_folds", "?"),
                r.get("uar", float("nan")),
                r.get("uar_std", 0.0),
                r.get("accuracy", float("nan")),
                r.get("accuracy_std", 0.0),
            )


def write_csv(path: Path, results: dict[int, dict[str, float | int]], vector: str) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "offset_k",
        "vector",
        "n",
        "n_scenes",
        "n_folds",
        "n_train",
        "n_test",
        "uar",
        "uar_std",
        "accuracy",
        "accuracy_std",
        "split_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for k in sorted(results):
            row = {"offset_k": k, "vector": vector, **results[k]}
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True, help="reintegration_detailed_fold*.json")
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--vector",
        type=str,
        default="masked",
        choices=("masked", "stable", "ghost"),
        help="Which hidden vector to decode",
    )
    p.add_argument("--num-classes", type=int, default=6, help="6 for IEMOCAP, 7 for MELD")
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits (scene-level)")
    p.add_argument("--max-offset", type=int, default=None, help="Only evaluate offsets <= this")
    p.add_argument(
        "--train-max-offset",
        type=int,
        default=None,
        help="If set with --test-min-offset, train on offsets <= this, test per offset",
    )
    p.add_argument(
        "--test-min-offset",
        type=int,
        default=None,
        help="Hold-out-offset generalization: test only on this offset and above",
    )
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--print-table", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_absence_rows(args.json, args.split)
    logging.info(
        "Loaded %s absence rows from %s [%s]",
        len(rows),
        args.json,
        args.split,
    )

    results = decode_by_offset(
        rows,
        vector=args.vector,
        n_splits=args.n_splits,
        max_offset=args.max_offset,
        train_max_offset=args.train_max_offset,
        test_min_offset=args.test_min_offset,
        random_state=args.random_state,
    )

    if args.print_table:
        print_table(results, num_classes=args.num_classes, vector=args.vector)

    payload = {
        "meta": {
            "json": str(args.json.resolve()),
            "split": args.split,
            "vector": args.vector,
            "num_classes": args.num_classes,
            "n_rows": len(rows),
            "chance_uar": CHANCE_UAR.get(args.num_classes),
        },
        "by_offset": {str(k): v for k, v in results.items()},
    }

    if args.out_csv:
        write_csv(args.out_csv, results, args.vector)
        logging.info("Wrote %s", args.out_csv.resolve())

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("Wrote %s", args.out_json.resolve())


if __name__ == "__main__":
    main()
