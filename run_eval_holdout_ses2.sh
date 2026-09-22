#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/aymie/Documents/UK_projects/masters-proj/my_extensions"
DATA="${ROOT}/reintegration/output/partition/holdout_ses_2"
export PYTHONPATH="${ROOT}"

cd "${ROOT}"
source /mnt/c/Users/aymie/Documents/UK_projects/masters-proj/fed-multimodal/wsl_venv/bin/activate

COND="mask_text_audio_live"
# COND="mask_audio_text_live"
# COND="mask_text_audio_zero"
# COND="text_zero_audio_live"
MASK="text"
# MASK="audio"
# EXTRA="" for live multimodal; for zero conditions add --en_audio_only or --en_text_only

OUT="${ROOT}/reintegration/output/all_txt_mask_audio_live/txt_mask_audio_live/holdout_session_2"

for FOLD in 1 2 3 4 5; do
  CKPT="${ROOT}/reintegration/trained_models/holdout_session_2/${COND}/fold${FOLD}/model.pt"
  echo "=== ${COND} fold ${FOLD} ==="

  python -m reintegration.train \
    --eval_only --ckpt_path "${CKPT}" \
    --run_name "${COND}" \
    --client_schedule_seed 2 \
    --mask_modality "${MASK}" \
    --dataset iemocap --data_dir "${DATA}" \
    --modality multimodal \
    --audio_feat mfcc --text_feat mobilebert \
    --fed_alg fed_avg --num_epochs 60 --local_epochs 1 \
    --sample_rate 0.5 --batch_size 16 --hid_size 128 --learning_rate 0.01 \
    --en_att --att_name fuse_base --availability_process markov \
    --reint_collect_fuse_attention \
    --reint_save_timestep_detail \
    --reint_ghost_pass \
    2>&1 | tee "${OUT}/eval_${COND}_fold${FOLD}.log"

  # Copy only the matching detailed JSON (written under masters-proj/result/...)
  RESULT_ROOT="/mnt/c/Users/aymie/Documents/UK_projects/masters-proj/result/fed_avg/iemocap/mfcc_mobilebert/fuse_base/hid128_le1_lr001_bs16_sr05_ep60/${COND}"
  cp "${RESULT_ROOT}/reintegration_detailed_fold${FOLD}.json" \
     "${OUT}/reintegration_detailed_fold${FOLD}.json"
done