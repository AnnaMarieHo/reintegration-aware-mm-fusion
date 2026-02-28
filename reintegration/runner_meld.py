"""
Entry points and utilities for MELD reintegration experiments that build on top
of the existing `fed_multimodal.experiment.meld.train` script.

Design goals:
- treat `fed_multimodal` as an installed library;
- keep the original MELD training script intact;
- layer in additional availability processes, logging, and analysis from this
  extension package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.model.mm_models import SERClassifier
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.server_trainer import Server

from .availability import (
    TwoStateMarkovParams,
    availability_history_counter,
    bucket_history_counters,
    reintegration_events,
    sample_two_state_markov,
)
from .metrics import ReintegrationAccumulator


@dataclass
class ReintegrationConfig:
    """
    Configuration for availability-history-based analysis on MELD.

    This is intentionally minimal: it focuses on the missingness schedule and
    analysis windows, while deferring model/backbone configuration to the
    existing `fed_multimodal` argument parser.
    """

    p_off_on: float = 0.05
    p_on_off: float = 0.05
    seed: int = 0


def run_meld_with_markov_availability(
    args,
    reint_cfg: Optional[ReintegrationConfig] = None,
) -> dict:
    """
    High-level scaffold for a MELD run under temporally correlated availability.

    Current status:
        - delegates backbone configuration, client sampling, and optimization
          entirely to `fed_multimodal`;
        - wires in placeholders for Markov availability and reintegration
          metrics via `ReintegrationAccumulator`.

    The concrete integration points (where Markov masks are sampled and how
    they are threaded through dataloaders and trainers) are left to be filled
    in as the next step.
    """
    if reint_cfg is None:
        reint_cfg = ReintegrationConfig()

    # 1. Construct MELD data manager using upstream code.
    dm = DataloadManager(args)
    dm.get_text_feat_path()
    dm.get_audio_feat_path()
    dm.get_meld_partition()

    # 2. Build model and server as usual.
    num_class = dm.get_num_class()
    model = SERClassifier(
        num_classes=num_class,
        audio_input_dim=dm.default_audio_feat_shape[-1],
        text_input_dim=dm.default_text_feat_shape[-1],
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name,
    )

    device = "cuda" if hasattr(args, "device") and args.device == "cuda" else "cpu"
    server = Server(
        args=args,
        device=device,
        model=model,
        dataload_manager=dm,
        client_trainer_cls=ClientFedAvg,
    )

    # 3. Placeholder: accumulator for reintegration metrics.
    reint_acc = ReintegrationAccumulator()

    # 4. Delegate training to the existing server, returning its result dict.
    #    Future work: hook into Server's evaluation to feed logits/labels and
    #    availability metadata into `reint_acc`.
    result = server.train()

    # 5. Compute (placeholder) reintegration summaries; at this scaffold stage
    #    the accumulator is empty, so these are effectively no-ops but define
    #    the JSON / dict shape that downstream plotting code can expect.
    result["reintegration_by_bucket"] = reint_acc.summary_by_bucket()
    result["reintegration_overall"] = reint_acc.summary_reintegration_vs_stable()

    return result

