#!/usr/bin/env bash
# Sweep the linear probe across all milestone checkpoints from a pretraining run.
# This generates the "linear-probe accuracy vs pretraining duration" curve
# the Phase 2 spec calls for.
#
# Usage:
#   ./scripts/probe_sweep.sh checkpoints/phase2_batch64_1.2m
#
# Or override the milestones being swept:
#   STEPS="200000 400000 600000 800000 1200000" ./scripts/probe_sweep.sh path/to/ckpts

set -euo pipefail

CKPT_DIR="${1:?usage: probe_sweep.sh <checkpoint-dir>}"
STEPS="${STEPS:-200000 400000 600000 800000 1200000}"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "Checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi

for step in ${STEPS}; do
    padded=$(printf "%08d" "${step}")
    ckpt="${CKPT_DIR}/milestone_step${padded}.pt"

    if [ ! -f "${ckpt}" ]; then
        echo "WARNING: milestone not found, skipping: ${ckpt}" >&2
        continue
    fi

    echo "=========================================="
    echo "Linear probe at step ${step}"
    echo "=========================================="

    # Only run the random-init control once (at the first milestone).
    if [ "${step}" = "$(echo ${STEPS} | cut -d' ' -f1)" ]; then
        run_random="true"
    else
        run_random="false"
    fi

    python -m scripts.linear_probe \
        probe.pretrained_checkpoint="${ckpt}" \
        probe.run_random_control="${run_random}" \
        log.run_name="probe_step${step}" \
        output_dir="results/linear_probe/step${step}"
done

echo "Sweep complete. Results in results/linear_probe/"
