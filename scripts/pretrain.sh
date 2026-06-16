#!/usr/bin/env bash
# Pretraining launcher.
#
# Usage:
#   ./scripts/pretrain.sh                          # single-GPU
#   NPROC=4 ./scripts/pretrain.sh                  # 4-GPU on one node
#   ./scripts/pretrain.sh model.mask_ratio=0.75    # Hydra override
#
# Environment variables consumed:
#   NPROC                    Number of GPUs (default 1)
#   NEUROSTREAM_CORPUS_INDEX Path to harmonised corpus sidecar JSON
#   NEUROSTREAM_CKPT_DIR     Where to save checkpoints
#   MLFLOW_TRACKING_URI      Inherited by the script if set

set -euo pipefail

NPROC="${NPROC:-1}"

# Sensible CUDA defaults — avoid the cuBLAS workspace warning on bf16.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Use the system Python's torchrun so we go through the same venv
# the project is installed into.
exec torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    -m neurostream.training.pretrain "$@"
