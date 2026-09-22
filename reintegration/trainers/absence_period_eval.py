"""
Full absence-period evaluation: from modality drop (mask 1→0) through every
absent timestep until reintegration (mask 0→1) or scene end.

Kept separate from the fixed-length reintegration recovery window so absence
spells of arbitrary length can be compared against post-return offsets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.metrics import recall_score

from reintegration.trainers.reintegration_metrics import (
    hidden_state_cosine_similarity,
    make_contrast_detail_row,
    per_class_window_metrics,
    record_contrast_timestep,
)


def _mean_by_offset(offset_lists: dict[int, list]) -> dict[int, float]:
    return {
        k: float(np.mean(v)) if v else float('nan')
        for k, v in sorted(offset_lists.items())
    }


def _find_absence_run_end(mask_np, t_drop: int, T: int) -> tuple[int, Optional[int]]:
    """
    Return (last_absent_index, t_reint).

    last_absent_index is inclusive. t_reint is the return index (0→1) if present
    before scene end, else None when the scene ends still absent.
    """
    t_k = t_drop
    while t_k < T and mask_np[t_k] == 0:
        t_k += 1
    last_absent = t_k - 1
    if last_absent < t_drop:
        return t_drop, None
    t_reint = int(t_k) if t_k < T and mask_np[t_k] == 1 else None
    return last_absent, t_reint


@dataclass
class AbsencePeriodAccumulator:
    """Collects per-timestep contrast metrics for full absence spells."""

    n_absence_events: int = 0
    run_lengths: list[int] = field(default_factory=list)
    delta_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    logp_gap_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    kl_forward_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    disagree_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    offset_labels: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    offset_preds_stable: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    offset_preds_masked: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    hidden_cosine_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    hidden_l2_by_offset: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    hidden_cosine_stable_ghost_by_offset: dict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    hidden_cosine_masked_ghost_by_offset: dict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    output_entropy_stable_by_offset: dict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    output_entropy_masked_by_offset: dict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    delta_output_entropy_by_offset: dict[int, list] = field(
        default_factory=lambda: defaultdict(list)
    )
    win_labels: list[int] = field(default_factory=list)
    win_preds_stable: list[int] = field(default_factory=list)
    win_preds_masked: list[int] = field(default_factory=list)
    timestep_detail: Optional[list[dict]] = None
    # _seen_window_t: set[int] = field(default_factory=set, repr=False)
    _seen_window_t: set[tuple[int, int]] = field(
        default_factory=set,
        repr=False
    )

    @classmethod
    def create(cls, save_timestep_detail: bool) -> AbsencePeriodAccumulator:
        acc = cls()
        if save_timestep_detail:
            acc.timestep_detail = []
        return acc

    def collect_scene(
        self,
        *,
        scene_batch_idx: int,
        T: int,
        mask_np,
        pred_s_np,
        pred_m_np,
        labels_np,
        log_p_s,
        log_p_m,
        scene_hidden_stable,
        scene_hidden_masked,
        out_ent_stable,
        out_ent_masked,
        save_hidden_vectors: bool,
        scene_hidden_ghost=None,
    ) -> None:
        """Scan one scene and record every full absence spell."""
        collect_ghost = scene_hidden_ghost is not None
        for t in range(1, T):
            if not (mask_np[t - 1] == 1 and mask_np[t] == 0):
                continue

            t_drop = t
            last_absent, t_reint = _find_absence_run_end(mask_np, t_drop, T)
            run_length = last_absent - t_drop + 1
            self.n_absence_events += 1
            self.run_lengths.append(run_length)

            for k, t_k in enumerate(range(t_drop, last_absent + 1)):
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
                    delta_by_offset=self.delta_by_offset,
                    logp_gap_by_offset=self.logp_gap_by_offset,
                    kl_forward_by_offset=self.kl_forward_by_offset,
                    disagree_by_offset=self.disagree_by_offset,
                    offset_labels=self.offset_labels,
                    offset_preds_stable=self.offset_preds_stable,
                    offset_preds_masked=self.offset_preds_masked,
                    hidden_cosine_by_offset=self.hidden_cosine_by_offset,
                    hidden_l2_by_offset=self.hidden_l2_by_offset,
                    output_entropy_stable_by_offset=self.output_entropy_stable_by_offset,
                    output_entropy_masked_by_offset=self.output_entropy_masked_by_offset,
                    delta_output_entropy_by_offset=self.delta_output_entropy_by_offset,
                )

                ghost_metrics = {}
                if collect_ghost:
                    cos_sg = hidden_state_cosine_similarity(
                        scene_hidden_stable[t_k], scene_hidden_ghost[t_k]
                    )
                    cos_mg = hidden_state_cosine_similarity(
                        scene_hidden_masked[t_k], scene_hidden_ghost[t_k]
                    )
                    self.hidden_cosine_stable_ghost_by_offset[k].append(cos_sg)
                    self.hidden_cosine_masked_ghost_by_offset[k].append(cos_mg)
                    ghost_metrics = {
                        'hidden_cosine_stable_ghost': cos_sg,
                        'hidden_cosine_masked_ghost': cos_mg,
                        'ghost_closer_to_stable_than_masked': float(
                            cos_sg > metrics['hidden_cosine_sim']
                        ),
                    }

                if self.timestep_detail is not None:
                    extra = {
                        'run_length': int(run_length),
                        't_reint': t_reint,
                        'is_last_in_run': bool(t_k == last_absent),
                    }
                    if ghost_metrics:
                        extra.update(ghost_metrics)
                    self.timestep_detail.append(
                        make_contrast_detail_row(
                            scene_batch_idx=scene_batch_idx,
                            t_event=t_drop,
                            t_event_key='t_drop',
                            k=k,
                            t_k=t_k,
                            labels_np=labels_np,
                            pred_s_np=pred_s_np,
                            pred_m_np=pred_m_np,
                            mask_np=mask_np,
                            metrics=metrics,
                            save_hidden_vectors=save_hidden_vectors,
                            scene_hidden_stable=scene_hidden_stable,
                            scene_hidden_masked=scene_hidden_masked,
                            extra=extra,
                            scene_hidden_ghost=scene_hidden_ghost if save_hidden_vectors else None,
                        )
                    )

                # if t_k not in self._seen_window_t:
                #     self._seen_window_t.add(t_k)
                #     self.win_labels.append(int(labels_np[t_k]))
                #     self.win_preds_stable.append(int(pred_s_np[t_k]))
                #     self.win_preds_masked.append(int(pred_m_np[t_k]))
                window_key = (int(scene_batch_idx), int(t_k))

                if window_key not in self._seen_window_t:
                    self._seen_window_t.add(window_key)

                    self.win_labels.append(int(labels_np[t_k]))
                    self.win_preds_stable.append(int(pred_s_np[t_k]))
                    self.win_preds_masked.append(int(pred_m_np[t_k]))

    def finalize(self, num_classes: int) -> dict[str, Any]:
        """Aggregate collected metrics into a JSON-serializable result block."""
        offsets = sorted(self.delta_by_offset.keys())

        if self.win_labels:
            uar_stable = recall_score(
                self.win_labels, self.win_preds_stable, average='macro', zero_division=0
            ) * 100
            uar_masked = recall_score(
                self.win_labels, self.win_preds_masked, average='macro', zero_division=0
            ) * 100
            delta_uar = uar_stable - uar_masked
        else:
            uar_stable = uar_masked = delta_uar = None

        absence_uar_by_offset = {}
        for k in offsets:
            if self.offset_labels[k]:
                uar_s = recall_score(
                    self.offset_labels[k],
                    self.offset_preds_stable[k],
                    average='macro',
                    zero_division=0,
                ) * 100
                uar_m = recall_score(
                    self.offset_labels[k],
                    self.offset_preds_masked[k],
                    average='macro',
                    zero_division=0,
                ) * 100
                absence_uar_by_offset[k] = {
                    'n': len(self.offset_labels[k]),
                    'uar_stable': float(uar_s),
                    'uar_masked': float(uar_m),
                    'delta_uar': float(uar_s - uar_m),
                }
            else:
                absence_uar_by_offset[k] = {
                    'n': 0,
                    'uar_stable': float('nan'),
                    'uar_masked': float('nan'),
                    'delta_uar': float('nan'),
                }

        per_class_uar_by_absence_offset = {}
        per_class_disagree_by_absence_offset = {}
        for k in offsets:
            class_metrics = per_class_window_metrics(
                self.offset_labels[k],
                self.offset_preds_stable[k],
                self.offset_preds_masked[k],
                num_classes,
            )
            per_class_uar_by_absence_offset[k] = {
                cls: {
                    'n': m['n'],
                    'uar_stable': m['uar_stable'],
                    'uar_masked': m['uar_masked'],
                    'delta_uar': m['delta_uar'],
                }
                for cls, m in class_metrics.items()
            }
            per_class_disagree_by_absence_offset[k] = {
                cls: {
                    'n': m['n'],
                    'disagree_rate': m['disagree_rate'],
                }
                for cls, m in class_metrics.items()
            }

        mean_delta = _mean_by_offset(self.delta_by_offset)
        run_lengths_arr = np.array(self.run_lengths, dtype=float) if self.run_lengths else None

        return {
            'n_absence_events': self.n_absence_events,
            'uar_stable_absence_period': uar_stable,
            'uar_masked_absence_period': uar_masked,
            'delta_uar_absence_period': delta_uar,
            'n_absence_period_timesteps': len(self.win_labels),
            'max_absence_offset_observed': max(offsets) if offsets else None,
            'mean_absence_run_length': float(run_lengths_arr.mean())
            if run_lengths_arr is not None and run_lengths_arr.size
            else None,
            'absence_run_lengths': list(self.run_lengths),
            'absence_uar_by_offset': absence_uar_by_offset,
            'mean_absence_delta_by_offset': mean_delta,
            'mean_absence_logp_gap_by_offset': _mean_by_offset(self.logp_gap_by_offset),
            'mean_absence_kl_forward_by_offset': _mean_by_offset(self.kl_forward_by_offset),
            'mean_absence_disagree_by_offset': _mean_by_offset(self.disagree_by_offset),
            'mean_absence_hidden_cosine_by_offset': _mean_by_offset(
                self.hidden_cosine_by_offset
            ),
            'mean_absence_hidden_l2_by_offset': _mean_by_offset(self.hidden_l2_by_offset),
            'mean_absence_hidden_cosine_stable_ghost_by_offset': _mean_by_offset(
                self.hidden_cosine_stable_ghost_by_offset
            ),
            'mean_absence_hidden_cosine_masked_ghost_by_offset': _mean_by_offset(
                self.hidden_cosine_masked_ghost_by_offset
            ),
            'absence_hidden_cosine_stable_ghost_by_offset': dict(
                self.hidden_cosine_stable_ghost_by_offset
            ),
            'absence_hidden_cosine_masked_ghost_by_offset': dict(
                self.hidden_cosine_masked_ghost_by_offset
            ),
            'mean_absence_output_entropy_stable_by_offset': _mean_by_offset(
                self.output_entropy_stable_by_offset
            ),
            'mean_absence_output_entropy_masked_by_offset': _mean_by_offset(
                self.output_entropy_masked_by_offset
            ),
            'mean_absence_delta_output_entropy_by_offset': _mean_by_offset(
                self.delta_output_entropy_by_offset
            ),
            'per_class_uar_by_absence_offset': per_class_uar_by_absence_offset,
            'per_class_disagree_by_absence_offset': per_class_disagree_by_absence_offset,
            'absence_delta_by_offset': dict(self.delta_by_offset),
            'absence_logp_gap_by_offset': dict(self.logp_gap_by_offset),
            'absence_kl_forward_by_offset': dict(self.kl_forward_by_offset),
            'absence_disagree_by_offset': dict(self.disagree_by_offset),
            'absence_hidden_cosine_by_offset': dict(self.hidden_cosine_by_offset),
            'absence_hidden_l2_by_offset': dict(self.hidden_l2_by_offset),
            'absence_output_entropy_stable_by_offset': dict(
                self.output_entropy_stable_by_offset
            ),
            'absence_output_entropy_masked_by_offset': dict(
                self.output_entropy_masked_by_offset
            ),
            'absence_delta_output_entropy_by_offset': dict(
                self.delta_output_entropy_by_offset
            ),
            'absence_timestep_detail': self.timestep_detail,
            # Legacy aliases (same values, clearer names preferred above)
            'uar_stable_absence_window': uar_stable,
            'uar_masked_absence_window': uar_masked,
            'delta_uar_absence_window': delta_uar,
            'n_absence_window_timesteps': len(self.win_labels),
        }


def log_absence_period_results(
    results: dict[str, Any],
    *,
    split_label: Optional[str] = None,
    reset_scene_hidden_each_step: bool = False,
    reset_scene_hidden_at_reintegration: bool = False,
    save_timestep_detail: bool = False,
) -> None:
    """Emit absence-period summary lines to the training log."""
    lp = f'[{split_label}] ' if split_label else ''
    # if reset_scene_hidden_each_step:
    #     lp += '[utt_encoder_recovery] '
    
    if reset_scene_hidden_each_step:
        lp += '[reset_each_step] '

    if reset_scene_hidden_at_reintegration:
        lp += '[reset_at_return] '

    n_events = results.get('n_absence_events', 0)
    n_ts = results.get('n_absence_period_timesteps', 0)
    logging.info(f'{lp}Absence period eval: n_events={n_events}, timesteps={n_ts}')

    if n_ts:
        logging.info(
            f'{lp}Absence period UAR: n={n_ts}, '
            f"UAR_stable={results['uar_stable_absence_period']:.2f}%, "
            f"UAR_masked={results['uar_masked_absence_period']:.2f}%, "
            f"delta={results['delta_uar_absence_period']:.2f}%"
        )

    run_lengths = results.get('absence_run_lengths') or []
    if run_lengths:
        logging.info(
            f'{lp}Absence run length: mean={np.mean(run_lengths):.2f}, '
            f'max={max(run_lengths)}, n_runs={len(run_lengths)}'
        )

    mean_delta = results.get('mean_absence_delta_by_offset') or {}
    absence_delta = results.get('absence_delta_by_offset') or {}
    if mean_delta:
        curve_str = ', '.join(
            f'+{k}:{mean_delta[k]:.4f} (n={len(absence_delta[k])})'
            for k in sorted(mean_delta.keys())
            if k in absence_delta
        )
        logging.info(f'{lp}Absence period curve (full run, offset from drop): {curve_str}')

    uar_by_off = results.get('absence_uar_by_offset') or {}
    if uar_by_off:
        uar_str = ', '.join(
            f"+{k}:delta={v['delta_uar']:.2f}% "
            f"(stable={v['uar_stable']:.2f}%, masked={v['uar_masked']:.2f}%, n={v['n']})"
            if v['n'] > 0
            else f"+{k}:delta=nan (n=0)"
            for k, v in sorted(uar_by_off.items())
        )
        logging.info(f'{lp}Absence period UAR by offset: {uar_str}')

    for label, key in (
        ('log-prob gap (nats)', 'mean_absence_logp_gap_by_offset'),
        ('KL(P_stable || P_masked) (nats)', 'mean_absence_kl_forward_by_offset'),
        ('argmax disagreement rate', 'mean_absence_disagree_by_offset'),
        ('scene GRU hidden cosine sim', 'mean_absence_hidden_cosine_by_offset'),
        (
            'scene GRU hidden cosine sim (stable vs ghost)',
            'mean_absence_hidden_cosine_stable_ghost_by_offset',
        ),
        (
            'scene GRU hidden cosine sim (masked vs ghost)',
            'mean_absence_hidden_cosine_masked_ghost_by_offset',
        ),
    ):
        series = results.get(key) or {}
        if series:
            s = ', '.join(f'+{k}:{series[k]:.4f}' for k in sorted(series.keys()))
            logging.info(f'{lp}Absence period {label}: {s}')

    if n_events == 0:
        logging.info(
            f'{lp}No absence events (no mask 1→0 transitions); absence period curves empty.'
        )

    if save_timestep_detail:
        detail = results.get('absence_timestep_detail') or []
        logging.info(f'{lp}Absence period timestep detail rows: {len(detail)}')
