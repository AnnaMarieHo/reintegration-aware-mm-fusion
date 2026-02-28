from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score


@dataclass
class ReintegrationAccumulator:
    """
    Lightweight accumulator for reintegration-window metrics.

    This is intended to sit alongside `fed_multimodal.trainers.evaluation.EvalMetric`
    without modifying it. Extension scripts can:

    - run the usual training/eval loop to obtain logits and labels,
    - attach per-sample metadata about availability history / reintegration,
    - feed that into this accumulator to compute:
        * accuracy stratified by reintegration bucket,
        * confidence stratified by reintegration bucket.
    """

    # Flattened over all batches / clients
    labels: List[int] = field(default_factory=list)
    preds: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    buckets: List[int] = field(default_factory=list)
    had_reintegration: List[bool] = field(default_factory=list)

    def add_batch(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        bucket_ids: np.ndarray,
        had_reintegration_flags: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            logits: array of shape (B, C) with unnormalized scores or
                log-probabilities per class.
            labels: array of shape (B,) with integer class labels.
            bucket_ids: array of shape (B,) giving the reintegration window
                bucket for each sample (e.g. derived from availability-history
                counters and their transforms).
            had_reintegration_flags: optional boolean array of shape (B,)
                indicating whether each sample experienced any OFF→ON event in
                its sequence. If omitted, this is inferred as
                `bucket_ids != 0` for summary-level metrics.
        """
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        bucket_ids = np.asarray(bucket_ids)

        assert logits.ndim == 2, "logits must be (B, C)"
        assert labels.shape[0] == logits.shape[0], "labels batch size mismatch"
        assert bucket_ids.shape[0] == logits.shape[0], "bucket_ids batch size mismatch"

        probs = _softmax(logits)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        self.labels.extend(labels.tolist())
        self.preds.extend(pred.tolist())
        self.confidences.extend(conf.tolist())
        self.buckets.extend(bucket_ids.tolist())

        if had_reintegration_flags is None:
            flags = bucket_ids != 0
        else:
            flags = np.asarray(had_reintegration_flags, dtype=bool)
            assert flags.shape[0] == logits.shape[0], "had_reintegration_flags batch size mismatch"
        self.had_reintegration.extend(flags.tolist())

    def summary_by_bucket(self) -> Dict[str, Dict[int, float]]:
        """
        Compute accuracy and mean confidence per reintegration bucket.

        Returns:
            {
              "acc": {bucket_id -> accuracy_in_bucket},
              "conf": {bucket_id -> mean_confidence_in_bucket},
            }
        """
        if not self.labels:
            return {"acc": {}, "conf": {}}

        labels = np.asarray(self.labels)
        preds = np.asarray(self.preds)
        conf = np.asarray(self.confidences)
        buckets = np.asarray(self.buckets)

        acc_by_bucket: Dict[int, float] = {}
        conf_by_bucket: Dict[int, float] = {}

        for b in np.unique(buckets):
            mask = buckets == b
            if not np.any(mask):
                continue
            acc_by_bucket[int(b)] = float(accuracy_score(labels[mask], preds[mask]) * 100.0)
            conf_by_bucket[int(b)] = float(conf[mask].mean())

        return {"acc": acc_by_bucket, "conf": conf_by_bucket}

    def summary_reintegration_vs_stable(self) -> Dict[str, float]:
        """
        Coarse-grained summary comparing sequences with and without observed
        OFF→ON reintegration events.

        Returns:
            {
              "acc_reint": ...,
              "acc_no_reint": ...,
              "conf_reint": ...,
              "conf_no_reint": ...,
            }
        """
        if not self.labels:
            return {
                "acc_reint": float("nan"),
                "acc_no_reint": float("nan"),
                "conf_reint": float("nan"),
                "conf_no_reint": float("nan"),
            }

        labels = np.asarray(self.labels)
        preds = np.asarray(self.preds)
        conf = np.asarray(self.confidences)
        flags = np.asarray(self.had_reintegration, dtype=bool)

        out: Dict[str, float] = {}

        mask_reint = flags
        mask_stable = ~flags

        if np.any(mask_reint):
            out["acc_reint"] = float(accuracy_score(labels[mask_reint], preds[mask_reint]) * 100.0)
            out["conf_reint"] = float(conf[mask_reint].mean())
        else:
            out["acc_reint"] = float("nan")
            out["conf_reint"] = float("nan")

        if np.any(mask_stable):
            out["acc_no_reint"] = float(accuracy_score(labels[mask_stable], preds[mask_stable]) * 100.0)
            out["conf_no_reint"] = float(conf[mask_stable].mean())
        else:
            out["acc_no_reint"] = float("nan")
            out["conf_no_reint"] = float("nan")

        return out


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(x, dtype=np.float32)
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=1, keepdims=True)

