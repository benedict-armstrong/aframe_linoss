#!/bin/bash

MODEL_ID=cusfflm4

echo "Starting Aframe run for model ID: ${MODEL_ID}"

# Export environment variables
export AFRAME_TRAIN_BACKGROUND_DIR=/home/barmstrong/aframe_new/data/bns/background_data_O3b
export AFRAME_TRAIN_WAVEFORMS_DIR=/home/barmstrong/aframe_new/data/bns/aframe
export AFRAME_TEST_BACKGROUND_DIR=/home/barmstrong/aframe_new/data/bns/background_data
export AFRAME_TEST_WAVEFORMS_DIR=/home/barmstrong/aframe_new/data/bns/aframe
export AFRAME_CONTAINER_ROOT=/fast/barmstrong/container/aframe

BASE_DIR=/fast/barmstrong/aframe_results/runs/${MODEL_ID}
export AFRAME_TRAIN_RUN_DIR=${BASE_DIR}/training
export AFRAME_CONDOR_DIR=${BASE_DIR}/condor
export AFRAME_RESULTS_DIR=${BASE_DIR}/results_aframe/
export AFRAME_TMPDIR=${AFRAME_RESULTS_DIR}/tmp

mkdir -p ${BASE_DIR}
mkdir -p ${AFRAME_RESULTS_DIR}


# launch pipeline; modify the gpus, workers etc. to suit your needs
# note that if you've made local code changes not in the containers
# you'll need to add the --dev flag!
# LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/sandbox.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.pipelines.sandbox.SandboxSV --workers 5 --gpus 0 --train-task SandboxSV
# LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/sandbox.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.tasks.export.ExportLocal --dev --workers 5 --gpus 0 --train-task ExportLocal

# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# PATH=/usr/bin:$PATH LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/base.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.tasks.infer.Infer --gpus 0,1,2,3,4,5,6 --dev --train-task Infer --log-level DEBUG #--print-status -1 # --print-deps -1

PATH=/usr/bin:$PATH LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/base.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.pipelines.sandbox.Sandbox --dev --train-task Infer  #--print-status -1 # --print-deps -1
