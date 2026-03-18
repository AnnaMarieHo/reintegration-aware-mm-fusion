#!/bin/bash
#SBATCH --job-name=iemocap_train
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --partition=V4V16_SKY32M192_L
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=gel_rhu285_f25cs660
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


source ~/.bashrc

echo "Available disk space:"
df -h /mnt/gpfs2_4m/scratch/ayho224/

conda activate $SCRATCH/reintegration-aware-mm-fusion/env
cd $SCRATCH/reintegration-aware-mm-fusion

export HF_HOME=/mnt/gpfs2_4m/scratch/ayho224/hf_cache
export HF_HUB_CACHE=/mnt/gpfs2_4m/scratch/ayho224/hf_cache
export TRANSFORMERS_CACHE=/mnt/gpfs2_4m/scratch/ayho224/hf_cache
export XDG_CACHE_HOME=/mnt/gpfs2_4m/scratch/ayho224/hf_cache
export HF_HUB_XET_CACHE_BASE=/mnt/gpfs2_4m/scratch/ayho224/hf_cache

module load ccs/cuda/12.2.0_535.54.03

echo "Running on: $(hostname)"
echo "Date: $(date)"
echo "Checking GPU status:"
nvidia-smi

python -c "import torch;
print('Torch version:', torch.__version__);
print('CUDA version:', torch.version.cuda);
print('CUDA available:', torch.cuda.is_available());
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

export PYTHONPATH=$PYTHONPATH:$SCRATCH/reintegration-aware-mm-fusion

python -m reintegration.train \
    --dataset iemocap \
    --data_dir reintegration/output \
    --modality multimodal \
    --audio_feat mfcc \
    --text_feat mobilebert \
    --fed_alg fed_avg \
    --num_epochs 60 \
    --local_epochs 1 \
    --sample_rate 1.0 \
    --batch_size 16 \
    --hid_size 128 \
    --learning_rate 0.01 \
    --en_att \
    --att_name fuse_base \
    --availability_process markov \

