from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from sklearn.metrics import accuracy_score

# Sample-level "reint vs no_reint" is invalid if no_reint group is too small
MIN_NO_REINT_FOR_SAMPLE_LEVEL = 200


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class SimpleReintegrationAccumulator:
    labels: List[int] = field(default_factory=list)
    preds: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    had_reintegration: List[bool] = field(default_factory=list)

    def add_batch(self, logits, labels, had_reint_flags):
        logits = np.asarray(logits)
        labels = np.asarray(labels).ravel()
        had_reint_flags = np.asarray(had_reint_flags, dtype=bool)

        probs = _softmax(logits)
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        self.labels.extend(labels.tolist())
        self.preds.extend(preds.tolist())
        self.confidences.extend(conf.tolist())
        self.had_reintegration.extend(had_reint_flags.tolist())

    def summary(self) -> Dict:
        """Returns acc/conf and sample/correct counts for reint vs no_reint (for denominator sanity checks)."""
        if not self.labels:
            return {}

        labels = np.asarray(self.labels)
        preds = np.asarray(self.preds)
        conf = np.asarray(self.confidences)
        flags = np.asarray(self.had_reintegration)

        mask_reint = flags
        mask_stable = ~flags

        n_reint = int(np.sum(mask_reint))
        n_no_reint = int(np.sum(mask_stable))
        correct_reint = int(np.sum((labels[mask_reint] == preds[mask_reint]))) if n_reint else 0
        correct_no_reint = int(np.sum((labels[mask_stable] == preds[mask_stable]))) if n_no_reint else 0

        result = {
            "num_samples_reint": n_reint,
            "num_samples_no_reint": n_no_reint,
            "num_correct_reint": correct_reint,
            "num_correct_no_reint": correct_no_reint,
        }

        if np.any(mask_reint):
            result["acc_reint"] = accuracy_score(labels[mask_reint], preds[mask_reint]) * 100
            result["conf_reint"] = float(conf[mask_reint].mean())
        else:
            result["acc_reint"] = float("nan")
            result["conf_reint"] = float("nan")

        if np.any(mask_stable):
            result["acc_no_reint"] = accuracy_score(labels[mask_stable], preds[mask_stable]) * 100
            result["conf_no_reint"] = float(conf[mask_stable].mean())
        else:
            result["acc_no_reint"] = float("nan")
            result["conf_no_reint"] = float("nan")

        return result


@dataclass
class EventLevelAccumulator:
    """Per-timestep (aux) accumulator: one (label, pred, conf, r_bucket) per valid timestep."""
    labels: List[int] = field(default_factory=list)
    preds: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    buckets: List[int] = field(default_factory=list)

    def add_batch_aux(
        self,
        aux_logits: np.ndarray,
        labels_per_sample: np.ndarray,
        mask_a_np: np.ndarray,
        len_a_np: np.ndarray,
        bucket_edges: tuple = (0, 1, 2, 3, 4, 8),
    ) -> None:
        from .availability import availability_history_counter, bucket_history_counters

        B, T_aux, C = aux_logits.shape
        probs = _softmax(aux_logits.reshape(-1, C))
        preds_flat = probs.argmax(axis=1)
        conf_flat = probs.max(axis=1)

        for b in range(B):
            len_a_raw = int(len_a_np[b])
            T_b = max(1, len_a_raw // 8)
            m = mask_a_np[b, :len_a_raw]
            if m.size == 0:
                continue
            m_red = downsample_mask_or(m, 8, T_b)
            m_red = np.asarray(m_red, dtype=bool).ravel()
            if len(m_red) == 0:
                continue
            r = availability_history_counter(m_red)
            buckets_b = bucket_history_counters(r, bucket_edges)
            label_b = int(labels_per_sample[b])
            for t in range(min(T_b, len(buckets_b))):
                idx = b * T_aux + t
                self.labels.append(label_b)
                self.preds.append(int(preds_flat[idx]))
                self.confidences.append(float(conf_flat[idx]))
                self.buckets.append(int(buckets_b[t]))

    def summary_by_bucket(self) -> Dict[str, Any]:
        if not self.labels:
            return {}
        labels = np.asarray(self.labels)
        preds = np.asarray(self.preds)
        conf = np.asarray(self.confidences)
        buckets = np.asarray(self.buckets)
        out = {}
        for bid in np.unique(buckets):
            mask = buckets == bid
            n = int(mask.sum())
            if n == 0:
                continue
            out[int(bid)] = {
                "n": n,
                "acc": float(accuracy_score(labels[mask], preds[mask]) * 100),
                "conf": float(conf[mask].mean()),
            }
        return out


def detect_reintegration(mask, length):
    """
    mask: (B, T)
    length: (B,)
    returns: (B,) boolean — True if this modality had any OFF→ON event in the sequence.
    """
    mask = mask.cpu().numpy()
    length = length.cpu().numpy()

    B = mask.shape[0]
    had_reint = np.zeros(B, dtype=bool)

    for b in range(B):
        L = int(length[b])
        m = mask[b, :L]
        if len(m) > 1:
            events = (m[:-1] == 0) & (m[1:] == 1)
            had_reint[b] = events.any()

    return had_reint


def detect_reintegration_multimodal(mask_a, mask_b, len_a, len_b):
    """
    Sample had reintegration if *either* modality had an OFF→ON event.
    mask_a, mask_b: (B, T); len_a, len_b: (B,).
    returns: (B,) boolean
    """
    had_reint_a = detect_reintegration(mask_a, len_a)
    had_reint_b = detect_reintegration(mask_b, len_b)
    return had_reint_a | had_reint_b


def count_off_on_events_raw(mask: np.ndarray, length: np.ndarray) -> int:
    """Count total OFF→ON events on raw mask across all samples. mask (B,T), length (B,)."""
    B = mask.shape[0]
    total = 0
    for b in range(B):
        L = int(length[b])
        m = mask[b, :L]
        if len(m) > 1:
            events = (m[:-1] == 0) & (m[1:] == 1)
            total += int(events.sum())
    return total


def downsample_mask_or(mask: np.ndarray, factor: int, target_len: int) -> np.ndarray:
    """
    Reduce 1D boolean mask by OR over consecutive windows. Matches model's
    downsample_mask_or (view(-1, factor).any(dim=1)) so aux_logits time and
    reduced mask are aligned. mask: (T,) or (B, T); returns (target_len,) or (B, target_len).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    B, T = mask.shape
    pad = (-T) % factor
    if pad:
        mask = np.concatenate([mask, np.zeros((B, pad), dtype=bool)], axis=1)
    mask = mask.reshape(B, -1, factor).any(axis=2)
    out = mask[:, :target_len]
    if squeeze:
        out = out[0]
    return out


def run_reintegration_eval(model, dataloader, device, multilabel: bool = False):
    """
    Run one eval pass: utterance-level acc/conf for samples with vs without
    reintegration. Reintegration = at least one OFF→ON event in *either*
    audio or text mask (multimodal). Returns summary + event_counts.
    No per-timestep/boundary logic yet—grouped by "any event in sample."
    """
    import torch

    model.eval()
    acc = SimpleReintegrationAccumulator()
    event_acc = EventLevelAccumulator()
    total_off_on_events_a = 0
    total_off_on_events_b = 0
    num_samples_with_reint = 0
    total_off_on_events_aux = 0
    bucket_edges = (0, 1, 2, 3, 4, 8)

    for x_a, x_b, l_a, l_b, y, mask_a, mask_b in dataloader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        l_a = l_a.to(device)
        l_b = l_b.to(device)
        mask_a = mask_a.to(device)
        mask_b = mask_b.to(device)

        with torch.no_grad():
            out = model(
                x_a.float(),
                x_b.float(),
                l_a,
                l_b,
                mask_a=mask_a,
                mask_b=mask_b,
                return_aux=True,
            )
        logits = out[0].cpu().numpy()
        aux_logits = out[2].cpu().numpy() if len(out) == 3 and out[2] is not None else None

        labels = y.cpu().numpy().ravel()
        had_reint = detect_reintegration_multimodal(mask_a, mask_b, l_a, l_b)

        if not multilabel:
            acc.add_batch(logits, labels, had_reint)

        mask_a_np = mask_a.cpu().numpy()
        mask_b_np = mask_b.cpu().numpy()
        l_a_np = l_a.cpu().numpy().ravel()
        l_b_np = l_b.cpu().numpy().ravel()
        total_off_on_events_a += count_off_on_events_raw(mask_a_np, l_a_np)
        total_off_on_events_b += count_off_on_events_raw(mask_b_np, l_b_np)
        num_samples_with_reint += int(had_reint.sum())

        # Event-level (aux time): r_t bucketing and accuracy per timestep
        if aux_logits is not None:
            event_acc.add_batch_aux(aux_logits, labels, mask_a_np, l_a_np, bucket_edges=bucket_edges)
            B = aux_logits.shape[0]
            for b in range(B):
                len_a_raw = int(l_a_np[b])
                T_aux_b = max(1, len_a_raw // 8)
                m = mask_a_np[b, :len_a_raw]
                if m.size > 1:
                    m_red = downsample_mask_or(m, 8, T_aux_b)
                    m_red = m_red.ravel()
                    if len(m_red) > 1:
                        events = (m_red[:-1] == 0) & (m_red[1:] == 1)
                        total_off_on_events_aux += int(events.sum())

    summary = acc.summary()
    n_no_reint = summary.get("num_samples_no_reint", 0)
    sample_level_valid = n_no_reint >= MIN_NO_REINT_FOR_SAMPLE_LEVEL
    summary["sample_level_reint_vs_no_reint_valid"] = sample_level_valid
    if not sample_level_valid and summary:
        summary["_warning"] = (
            f"SKIPPING sample-level reint vs no_reint as headline: "
            f"control group too small (n_no_reint={n_no_reint}, need >={MIN_NO_REINT_FOR_SAMPLE_LEVEL})."
        )

    summary["event_counts"] = {
        "total_off_on_events_audio": total_off_on_events_a,
        "total_off_on_events_text": total_off_on_events_b,
        "total_off_on_events": total_off_on_events_a + total_off_on_events_b,
        "num_samples_with_reint": num_samples_with_reint,
    }
    summary["event_counts_aux"] = {"total_off_on_events": total_off_on_events_aux}

    event_by_bucket = event_acc.summary_by_bucket()
    if event_by_bucket:
        summary["event_level_by_bucket"] = {str(k): v for k, v in event_by_bucket.items()}

    return summary