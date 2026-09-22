"""
Standalone ghost hidden-state probe: stable vs masked vs ghost (post-RNN null) passes.

Runs three forward passes per scene and records absence-period hidden cosine metrics:
  - cos(h_stable, h_masked)   — existing degradation measure
  - cos(h_stable, h_ghost)    — distance to stable under ghost encoding
  - cos(h_masked, h_ghost)    — how much ghost differs from standard masked pass

Example:
  python -m reintegration.scripts.run_ghost_hidden_probe \\
      --ckpt reintegration/trained_models/holdout_session_3/mask_text_audio_live/fold1/model.pt \\
      --data_dir reintegration/output/partition/holdout_ses_3 \\
      --split test --mask_modality text --out ghost_probe_fold1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from reintegration.constants import constants
from reintegration.dataloader.dataload_manager import DataloadManager
from reintegration.model.mm_models import SERClassifier, SceneGRUWrapper
from reintegration.trainers.absence_period_eval import AbsencePeriodAccumulator, log_absence_period_results
from reintegration.trainers.server_trainer import sanitize_for_json

logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def infer_fold_from_ckpt(path: Path) -> int | None:
    m = re.search(r"fold(\d+)", str(path))
    return int(m.group(1)) if m else None


def build_scene_model(
    *,
    mask_modality: str,
    hid_size: int,
    att: bool,
    att_name: str,
    dataset: str,
) -> SceneGRUWrapper:
    utterance_encoder = SERClassifier(
        num_classes=constants.num_class_dict[dataset],
        audio_input_dim=constants.feature_len_dict["mfcc"],
        text_input_dim=constants.feature_len_dict["mobilebert"],
        d_hid=hid_size,
        n_filters=32,
        en_att=att,
        att_name=att_name,
        d_head=6,
        audio_only=False,
        text_only=False,
    )
    return SceneGRUWrapper(
        utterance_encoder=utterance_encoder,
        num_classes=constants.num_class_dict[dataset],
        d_hid=hid_size,
        mask_modality=mask_modality,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()


def build_dataloader(args, split: str):
    dm = DataloadManager(args)
    dm.get_text_feat_path()
    partition_path = Path(args.data_dir).joinpath("partition", args.dataset, "partition.json")
    with open(partition_path, encoding="utf-8") as f:
        partition = json.load(f)

    dm.get_client_ids()
    if split not in dm.client_ids:
        raise KeyError(f"split {split!r} not in client ids: {dm.client_ids}")

    audio_feat_dict = dm.load_audio_feat(client_id=split)
    text_feat_dict = dm.load_text_feat(client_id=split)
    scenes = partition[str(split)]

    return dm.set_scene_dataloader(
        scenes=scenes,
        audio_feat_dict=audio_feat_dict,
        text_feat_dict=text_feat_dict,
        default_feat_shape_a=np.array([1000, constants.feature_len_dict["mfcc"]]),
        default_feat_shape_b=np.array([10, constants.feature_len_dict["mobilebert"]]),
        p_stay_absent=0.7,
        p_stay_present=0.75,
        shuffle=False,
        apply_mask=True,
    )


def run_ghost_probe(
    model: SceneGRUWrapper,
    dataloader,
    device: torch.device,
    *,
    save_timestep_detail: bool,
) -> dict:
    num_classes = model.num_classes
    absence_acc = AbsencePeriodAccumulator.create(save_timestep_detail)
    scene_batch_idx = -1

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="ghost probe"):
            scene_batch_idx += 1
            (
                scene_x_a,
                scene_x_b,
                scene_len_a,
                scene_len_b,
                scene_labels,
                scene_mask,
            ) = batch_data

            scene_labels = scene_labels.to(device)
            scene_mask = scene_mask.to(device)
            T = scene_labels.shape[0]
            ones_mask = torch.ones(T, device=device, dtype=torch.long)

            preds_stable, scene_hidden_stable = model(
                scene_x_a,
                scene_x_b,
                scene_len_a,
                scene_len_b,
                ones_mask,
                device,
                encoding_mode="standard",
            )
            preds_masked, scene_hidden_masked = model(
                scene_x_a,
                scene_x_b,
                scene_len_a,
                scene_len_b,
                scene_mask,
                device,
                encoding_mode="standard",
            )
            _, scene_hidden_ghost = model(
                scene_x_a,
                scene_x_b,
                scene_len_a,
                scene_len_b,
                scene_mask,
                device,
                encoding_mode="ghost",
            )

            mask_np = scene_mask.cpu().numpy()
            labels_np = scene_labels.cpu().numpy()
            pred_s_np = preds_stable.argmax(dim=-1).cpu().numpy()
            pred_m_np = preds_masked.argmax(dim=-1).cpu().numpy()
            log_p_s = torch.nn.functional.log_softmax(preds_stable, dim=-1).detach().cpu().numpy()
            log_p_m = torch.nn.functional.log_softmax(preds_masked, dim=-1).detach().cpu().numpy()

            absence_acc.collect_scene(
                scene_batch_idx=scene_batch_idx,
                T=T,
                mask_np=mask_np,
                pred_s_np=pred_s_np,
                pred_m_np=pred_m_np,
                labels_np=labels_np,
                log_p_s=log_p_s,
                log_p_m=log_p_m,
                scene_hidden_stable=scene_hidden_stable,
                scene_hidden_masked=scene_hidden_masked,
                out_ent_stable=torch.zeros(T),
                out_ent_masked=torch.zeros(T),
                save_hidden_vectors=save_timestep_detail,
                scene_hidden_ghost=scene_hidden_ghost,
            )

    return absence_acc.finalize(num_classes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset", type=str, default="iemocap")
    p.add_argument("--audio_feat", type=str, default="mfcc")
    p.add_argument("--text_feat", type=str, default="mobilebert")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mask_modality", type=str, default="text", choices=("audio", "text"))
    p.add_argument("--hid_size", type=int, default=128)
    p.add_argument("--att", action="store_true", default=True)
    p.add_argument("--no_att", action="store_false", dest="att")
    p.add_argument("--att_name", type=str, default="fuse_base")
    p.add_argument("--client_schedule_seed", type=int, default=None)
    p.add_argument("--availability_process", type=str, default="markov")
    p.add_argument("--save_timestep_detail", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    fold = args.fold or infer_fold_from_ckpt(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("Checkpoint: %s", args.ckpt)

    model = build_scene_model(
        mask_modality=args.mask_modality,
        hid_size=args.hid_size,
        att=args.att,
        att_name=args.att_name,
        dataset=args.dataset,
    ).to(device)
    load_checkpoint(model, args.ckpt, device)

    dataloader = build_dataloader(args, args.split)
    results = run_ghost_probe(
        model,
        dataloader,
        device,
        save_timestep_detail=args.save_timestep_detail,
    )

    log_absence_period_results(
        results,
        split_label=args.split,
        save_timestep_detail=args.save_timestep_detail,
    )

    payload = {
        "meta": {
            "fold": fold,
            "split": args.split,
            "dataset": args.dataset,
            "mask_modality": args.mask_modality,
            "ckpt": str(args.ckpt.resolve()),
            "encoding_modes": ["standard_stable", "standard_masked", "ghost"],
        },
        "summary": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(sanitize_for_json(payload), indent=4),
        encoding="utf-8",
    )
    logging.info("Wrote %s", args.out.resolve())

    cos_sm = results.get("mean_absence_hidden_cosine_by_offset") or {}
    cos_sg = results.get("mean_absence_hidden_cosine_stable_ghost_by_offset") or {}
    if cos_sm and cos_sg:
        for k in sorted(set(cos_sm) & set(cos_sg), key=lambda x: int(x)):
            logging.info(
                "offset +%s: cos(stable,masked)=%.4f cos(stable,ghost)=%.4f "
                "ghost_closer=%s",
                k,
                cos_sm[k],
                cos_sg[k],
                cos_sg[k] > cos_sm[k],
            )


if __name__ == "__main__":
    main()
