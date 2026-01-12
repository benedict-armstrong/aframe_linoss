#!/bin/bash
# Export environment variables
export AFRAME_TRAIN_BACKGROUND_DIR=/home/barmstrong/aframe_new/data/bns/background_data_O3b
export AFRAME_TRAIN_WAVEFORMS_DIR=/home/barmstrong/aframe_new/data/bns/aframe
export AFRAME_TEST_BACKGROUND_DIR=/home/barmstrong/aframe_new/data/bns/background_data
export AFRAME_TEST_WAVEFORMS_DIR=/home/barmstrong/aframe_new/data/bns/aframe
export AFRAME_TRAIN_RUN_DIR=/lustre/home/barmstrong/aframe_new/runs/training
export AFRAME_CONDOR_DIR=/lustre/home/barmstrong/aframe_new/runs/condor
export AFRAME_RESULTS_DIR=/lustre/home/barmstrong/aframe_new/runs/results_aframe
export AFRAME_TMPDIR=/lustre/home/barmstrong/aframe_new/runs/results_aframe/tmp
export AFRAME_CONTAINER_ROOT=/fast/barmstrong/container/aframe

# launch pipeline; modify the gpus, workers etc. to suit your needs
# note that if you've made local code changes not in the containers
# you'll need to add the --dev flag!
# LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/sandbox.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.pipelines.sandbox.SandboxSV --workers 5 --gpus 0 --train-task SandboxSV
# LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/sandbox.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.tasks.export.ExportLocal --dev --workers 5 --gpus 0 --train-task ExportLocal

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
PATH=/usr/bin:$PATH LAW_CONFIG_FILE=/lustre/home/barmstrong/aframe_new/runs/base.cfg uv run --directory /lustre/home/barmstrong/aframe_new law run aframe.tasks.infer.Infer --gpus 0,1,2,3 --dev --train-task Infer --log-level DEBUG #--print-status -1 # --print-deps -1
