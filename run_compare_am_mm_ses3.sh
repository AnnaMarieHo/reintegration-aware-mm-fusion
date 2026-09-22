#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions"
DATA="${ROOT}/reintegration/output/partition/holdout_ses_3"
export PYTHONPATH="${ROOT}"

cd "${ROOT}"
source /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/fed-multimodal/venv/bin/activate

# Multimodal checkpoint (text masked at eval for this study)
MM_COND="mask_text_audio_live"
# Audio-only specialist shadow model
AM_COND="text_zero_audio_live"
MASK="text"

OUT="${ROOT}/reintegration/output/all_txt_mask_audio_live/txt_mask_audio_live/holdout_session_3"

for FOLD in 1 2 3 4 5; do
  MM_CKPT="${ROOT}/reintegration/trained_models/holdout_session_3/${MM_COND}/fold${FOLD}/model.pt"
  AM_CKPT="${ROOT}/reintegration/trained_models/holdout_session_3/${AM_COND}/fold${FOLD}/model.pt"
  echo "=== AM vs MM divergence | MM=${MM_COND} AM=${AM_COND} fold ${FOLD} ==="

  python -m reintegration.scripts.compare_am_mm_divergence \
    --mm_ckpt "${MM_CKPT}" \
    --am_ckpt "${AM_CKPT}" \
    --data_dir "${DATA}" \
    --split test \
    --mask_modality "${MASK}" \
    --client_schedule_seed 2 \
    --save_timestep_detail \
    --fold "${FOLD}" \
    --out "${OUT}/am_mm_divergence_fold${FOLD}.json" \
    2>&1 | tee "${OUT}/compare_am_mm_${MM_COND}_fold${FOLD}.log"
done
