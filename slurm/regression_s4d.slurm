#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=regression_s4d
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_s4d/logs/slurm_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_s4d/logs/slurm_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss
BACKGROUND_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3b_H1_L1_4096Hz
RUN_DIR=${REPO}/runs/regression_s4d

mkdir -p "${RUN_DIR}/logs"

source ~/.bash_profile

export WANDB_DIR="${RUN_DIR}"

cd ${REPO}/projects/train
uv run --no-sync python -m train.regression_cli fit \
    --config configs/regression_s4d.yaml \
    --data.init_args.background_dir "${BACKGROUND_DIR}" \
    --trainer.default_root_dir "${RUN_DIR}"
