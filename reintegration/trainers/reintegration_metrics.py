"""Shared stable-vs-masked contrast metrics for reintegration evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def hidden_state_cosine_similarity(h_stable: torch.Tensor, h_masked: torch.Tensor) -> float:
    """Cosine similarity between two scene-GRU hidden vectors."""
    return float(
        F.cosine_similarity(
            h_stable.unsqueeze(0),
            h_masked.unsqueeze(0),
            dim=1,
        ).item()
    )


def hidden_state_l2_distance(h_stable: torch.Tensor, h_masked: torch.Tensor) -> float:
    """L2 distance between two scene-GRU hidden vectors."""
    return float(torch.norm(h_stable - h_masked).item())


def output_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits) per row (nats). logits: (..., num_classes)."""
    probs = F.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)


def per_class_window_metrics(
    labels: list,
    preds_stable: list,
    preds_masked: list,
    num_classes: int,
) -> dict:
    """Per-class recall and argmax disagreement within one recovery-window slice."""
    out = {}
    for c in range(num_classes):
        idx = [i for i, y in enumerate(labels) if y == c]
        n = len(idx)
        if n == 0:
            out[str(c)] = {
                'n': 0,
                'uar_stable': float('nan'),
                'uar_masked': float('nan'),
                'delta_uar': float('nan'),
                'disagree_rate': float('nan'),
            }
            continue
        uar_s = sum(1 for i in idx if preds_stable[i] == c) / n * 100
        uar_m = sum(1 for i in idx if preds_masked[i] == c) / n * 100
        disagree = sum(1 for i in idx if preds_stable[i] != preds_masked[i]) / n
        out[str(c)] = {
            'n': n,
            'uar_stable': float(uar_s),
            'uar_masked': float(uar_m),
            'delta_uar': float(uar_s - uar_m),
            'disagree_rate': float(disagree),
        }
    return out


def record_contrast_timestep(
    k: int,
    t_k: int,
    *,
    pred_s_np,
    pred_m_np,
    labels_np,
    log_p_s,
    log_p_m,
    scene_hidden_stable,
    scene_hidden_masked,
    out_ent_stable,
    out_ent_masked,
    delta_by_offset,
    logp_gap_by_offset,
    kl_forward_by_offset,
    disagree_by_offset,
    offset_labels,
    offset_preds_stable,
    offset_preds_masked,
    hidden_cosine_by_offset,
    hidden_l2_by_offset,
    output_entropy_stable_by_offset,
    output_entropy_masked_by_offset,
    delta_output_entropy_by_offset,
) -> dict:
    """Append stable-vs-masked metrics for one offset k; return scalar bundle."""
    correct_s = int(pred_s_np[t_k] == labels_np[t_k])
    correct_m = int(pred_m_np[t_k] == labels_np[t_k])
    delta_by_offset[k].append(correct_s - correct_m)

    y_idx = int(labels_np[t_k])
    logp_gap = float(log_p_s[t_k, y_idx] - log_p_m[t_k, y_idx])
    logp_gap_by_offset[k].append(logp_gap)

    p_row = np.exp(log_p_s[t_k])
    kl_f = float(np.sum(p_row * (log_p_s[t_k] - log_p_m[t_k])))
    kl_forward_by_offset[k].append(kl_f)

    disagree_by_offset[k].append(float(pred_s_np[t_k] != pred_m_np[t_k]))
    offset_labels[k].append(int(labels_np[t_k]))
    offset_preds_stable[k].append(int(pred_s_np[t_k]))
    offset_preds_masked[k].append(int(pred_m_np[t_k]))

    h_cos = hidden_state_cosine_similarity(
        scene_hidden_stable[t_k], scene_hidden_masked[t_k]
    )
    h_l2 = hidden_state_l2_distance(
        scene_hidden_stable[t_k], scene_hidden_masked[t_k]
    )
    ent_s = float(out_ent_stable[t_k].item())
    ent_m = float(out_ent_masked[t_k].item())
    ent_delta = ent_m - ent_s
    hidden_cosine_by_offset[k].append(h_cos)
    hidden_l2_by_offset[k].append(h_l2)
    output_entropy_stable_by_offset[k].append(ent_s)
    output_entropy_masked_by_offset[k].append(ent_m)
    delta_output_entropy_by_offset[k].append(ent_delta)

    return {
        'correct_s': correct_s,
        'correct_m': correct_m,
        'logp_gap': logp_gap,
        'kl_forward': kl_f,
        'hidden_cosine_sim': h_cos,
        'hidden_l2_distance': h_l2,
        'output_entropy_stable': ent_s,
        'output_entropy_masked': ent_m,
        'delta_output_entropy': ent_delta,
    }


def make_contrast_detail_row(
    *,
    scene_batch_idx: int,
    t_event: int,
    t_event_key: str,
    k: int,
    t_k: int,
    labels_np,
    pred_s_np,
    pred_m_np,
    mask_np,
    metrics: dict,
    save_hidden_vectors: bool,
    scene_hidden_stable,
    scene_hidden_masked,
    extra: dict | None = None,
    scene_hidden_ghost=None,
) -> dict:
    row = {
        'scene_batch_idx': int(scene_batch_idx),
        t_event_key: int(t_event),
        'offset_k': int(k),
        't_abs': int(t_k),
        'y_true': int(labels_np[t_k]),
        'pred_stable': int(pred_s_np[t_k]),
        'pred_masked': int(pred_m_np[t_k]),
        'correct_stable': int(metrics['correct_s']),
        'correct_masked': int(metrics['correct_m']),
        'delta_correct': int(metrics['correct_s'] - metrics['correct_m']),
        'logp_gap': float(metrics['logp_gap']),
        'kl_forward': float(metrics['kl_forward']),
        'disagree': float(pred_s_np[t_k] != pred_m_np[t_k]),
        'mask_t': int(mask_np[t_k]),
        'hidden_cosine_sim': float(metrics['hidden_cosine_sim']),
        'hidden_l2_distance': float(metrics['hidden_l2_distance']),
        'output_entropy_stable': float(metrics['output_entropy_stable']),
        'output_entropy_masked': float(metrics['output_entropy_masked']),
        'delta_output_entropy': float(metrics['delta_output_entropy']),
    }
    if save_hidden_vectors:
        row['hidden_stable'] = scene_hidden_stable[t_k].detach().cpu().tolist()
        row['hidden_masked'] = scene_hidden_masked[t_k].detach().cpu().tolist()
        if scene_hidden_ghost is not None:
            row['hidden_ghost'] = scene_hidden_ghost[t_k].detach().cpu().tolist()
    if extra:
        row.update(extra)
    return row


def collect_recovery_window(
    *,
    t_event: int,
    T: int,
    mask_np,
    window: int,
    stop_when_mask_value: int,
    pred_s_np,
    pred_m_np,
    labels_np,
    log_p_s,
    log_p_m,
    scene_hidden_stable,
    scene_hidden_masked,
    out_ent_stable,
    out_ent_masked,
    delta_by_offset,
    logp_gap_by_offset,
    kl_forward_by_offset,
    disagree_by_offset,
    offset_labels,
    offset_preds_stable,
    offset_preds_masked,
    hidden_cosine_by_offset,
    hidden_l2_by_offset,
    output_entropy_stable_by_offset,
    output_entropy_masked_by_offset,
    delta_output_entropy_by_offset,
    timestep_detail,
    t_event_key: str,
    scene_batch_idx: int,
    save_timestep_detail: bool,
    seen_window_t: set,
    win_labels: list,
    win_preds_stable: list,
    win_preds_masked: list,
    extra=None,
) -> None:
    """Record metrics at offsets +0..+window while mask stays != stop_when_mask_value."""
    for k in range(window + 1):
        t_k = t_event + k
        if t_k >= T:
            break
        if k > 0 and mask_np[t_k] == stop_when_mask_value:
            break

        metrics = record_contrast_timestep(
            k,
            t_k,
            pred_s_np=pred_s_np,
            pred_m_np=pred_m_np,
            labels_np=labels_np,
            log_p_s=log_p_s,
            log_p_m=log_p_m,
            scene_hidden_stable=scene_hidden_stable,
            scene_hidden_masked=scene_hidden_masked,
            out_ent_stable=out_ent_stable,
            out_ent_masked=out_ent_masked,
            delta_by_offset=delta_by_offset,
            logp_gap_by_offset=logp_gap_by_offset,
            kl_forward_by_offset=kl_forward_by_offset,
            disagree_by_offset=disagree_by_offset,
            offset_labels=offset_labels,
            offset_preds_stable=offset_preds_stable,
            offset_preds_masked=offset_preds_masked,
            hidden_cosine_by_offset=hidden_cosine_by_offset,
            hidden_l2_by_offset=hidden_l2_by_offset,
            output_entropy_stable_by_offset=output_entropy_stable_by_offset,
            output_entropy_masked_by_offset=output_entropy_masked_by_offset,
            delta_output_entropy_by_offset=delta_output_entropy_by_offset,
        )

        if timestep_detail is not None:
            timestep_detail.append(
                make_contrast_detail_row(
                    scene_batch_idx=scene_batch_idx,
                    t_event=t_event,
                    t_event_key=t_event_key,
                    k=k,
                    t_k=t_k,
                    labels_np=labels_np,
                    pred_s_np=pred_s_np,
                    pred_m_np=pred_m_np,
                    mask_np=mask_np,
                    metrics=metrics,
                    save_hidden_vectors=save_timestep_detail,
                    scene_hidden_stable=scene_hidden_stable,
                    scene_hidden_masked=scene_hidden_masked,
                    extra=extra,
                )
            )

        if t_k not in seen_window_t:
            seen_window_t.add(t_k)
            win_labels.append(int(labels_np[t_k]))
            win_preds_stable.append(int(pred_s_np[t_k]))
            win_preds_masked.append(int(pred_m_np[t_k]))
