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

def reintegration_metadata_from_masks(
    mask_a,
    mask_b,
    len_a,
    len_b,
    bucket_edges: Sequence[int] = (0, 1, 2, 3, 4, 8),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample reintegration flags and bucket IDs from availability masks.

    Args:
        mask_a: (B, T_a) bool tensor or array; True = modality ON.
        mask_b: (B, T_b) bool tensor or array.
        len_a, len_b: (B,) int tensor or array; valid length per sample.
        bucket_edges: edges for bucketing r (steps since last reintegration).

    Returns:
        had_reintegration: (B,) bool; True if sequence had any OFF→ON event.
        bucket_ids: (B,) int; bucket index per sample (from r at last valid step).
    """
    from .availability import (
        availability_history_counter,
        bucket_history_counters,
        reintegration_events,
    )

    if hasattr(mask_a, "cpu"):
        mask_a = mask_a.cpu().numpy()
    if hasattr(mask_b, "cpu"):
        mask_b = mask_b.cpu().numpy()
    if hasattr(len_a, "cpu"):
        len_a = len_a.cpu().numpy()
    if hasattr(len_b, "cpu"):
        len_b = len_b.cpu().numpy()
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    len_a = np.asarray(len_a, dtype=np.int32).ravel()
    len_b = np.asarray(len_b, dtype=np.int32).ravel()

    B = mask_a.shape[0]
    had_reintegration = np.zeros(B, dtype=bool)
    bucket_ids = np.zeros(B, dtype=np.int32)

    for b in range(B):
        L_a = int(len_a[b])
        L_b = int(len_b[b])
        ma = mask_a[b, :L_a] if L_a > 0 else np.zeros(0, dtype=bool)
        mb = mask_b[b, :L_b] if L_b > 0 else np.zeros(0, dtype=bool)
        events_a = reintegration_events(ma)
        events_b = reintegration_events(mb)
        had_reintegration[b] = bool(events_a.any() or events_b.any())
        r_a = availability_history_counter(ma)
        r_b = availability_history_counter(mb)
        r_at_end = 0
        if len(r_a) > 0:
            r_at_end = max(r_at_end, int(r_a[-1]))
        if len(r_b) > 0:
            r_at_end = max(r_at_end, int(r_b[-1]))
        buck = bucket_history_counters(np.array([r_at_end], dtype=np.int32), bucket_edges)
        bucket_ids[b] = int(buck[0])

    return had_reintegration, bucket_ids


def run_reintegration_eval(
    model,
    dataloader,
    device,
    multilabel: bool = False,
    bucket_edges: Sequence[int] = (0, 1, 2, 3, 4, 8),
):
    """
    Run one eval pass and compute reintegration-window metrics.

    Args:
        model: model with forward(x_a, x_b, l_a, l_b, mask_a=..., mask_b=...).
        dataloader: yields (x_a, x_b, l_a, l_b, y, mask_a, mask_b).
        device: torch device.
        multilabel: if True, skip reintegration accumulator (or adapt as needed).
        bucket_edges: for bucketing availability-history counter.

    Returns:
        {
          "by_bucket": {"acc": {bid: ...}, "conf": {bid: ...}},
          "overall": {"acc_reint": ..., "acc_no_reint": ..., ...},
          "event_counts": {"total_off_on_events": ..., "num_samples_with_reint": ...},
        }
    """
    import torch
    from .availability import reintegration_events

    model.eval()
    acc = ReintegrationAccumulator()
    total_off_on_events = 0
    num_samples_with_reint = 0

    for batch_data in dataloader:
        x_a, x_b, l_a, l_b, y, mask_a, mask_b = batch_data
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        l_a = l_a.to(device)
        l_b = l_b.to(device)
        mask_a = mask_a.to(device)
        mask_b = mask_b.to(device)
        with torch.no_grad():
            outputs, _ = model(
                x_a.float(), x_b.float(), l_a, l_b,
                mask_a=mask_a, mask_b=mask_b,
            )
        if multilabel:
            continue
        logits = outputs.detach().cpu().numpy()
        labels = y.detach().cpu().numpy().ravel()
        had_reint, bucket_ids = reintegration_metadata_from_masks(
            mask_a, mask_b, l_a, l_b, bucket_edges=bucket_edges
        )
        acc.add_batch(logits, labels, bucket_ids, had_reintegration_flags=had_reint)
        for b in range(labels.shape[0]):
            La = int(l_a[b].item())
            Lb = int(l_b[b].item())
            ma = mask_a[b, :La].cpu().numpy() if La > 0 else np.zeros(0, dtype=bool)
            mb = mask_b[b, :Lb].cpu().numpy() if Lb > 0 else np.zeros(0, dtype=bool)
            total_off_on_events += reintegration_events(ma).sum() + reintegration_events(mb).sum()
            if had_reint[b]:
                num_samples_with_reint += 1

    if not acc.labels:
        return {
            "by_bucket": {"acc": {}, "conf": {}},
            "overall": {
                "acc_reint": float("nan"),
                "acc_no_reint": float("nan"),
                "conf_reint": float("nan"),
                "conf_no_reint": float("nan"),
            },
            "event_counts": {"total_off_on_events": 0, "num_samples_with_reint": 0},
        }
    by_bucket = acc.summary_by_bucket()
    by_bucket_serializable = {
        "acc": {str(k): v for k, v in by_bucket["acc"].items()},
        "conf": {str(k): v for k, v in by_bucket["conf"].items()},
    }
    return {
        "by_bucket": by_bucket_serializable,
        "overall": acc.summary_reintegration_vs_stable(),
        "event_counts": {
            "total_off_on_events": int(total_off_on_events),
            "num_samples_with_reint": int(num_samples_with_reint),
        },
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = np.asarray(x, dtype=np.float32)
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=1, keepdims=True)

