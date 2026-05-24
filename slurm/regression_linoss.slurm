#!/bin/bash
#SBATCH --job-name=regression_linoss
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/regression_linoss_%j.out
#SBATCH --error=logs/regression_linoss_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss
BACKGROUND_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_linoss/background/train
RUN_DIR=${REPO}/runs/regression_linoss

mkdir -p "${RUN_DIR}" logs

source ~/.bash_profile

cd ${REPO}/projects/train
uv run --no-sync python -m train.regression_cli fit \
    --config configs/regression_linoss.yaml \
    --data.init_args.background_dir "${BACKGROUND_DIR}" \
    --trainer.default_root_dir "${RUN_DIR}"
