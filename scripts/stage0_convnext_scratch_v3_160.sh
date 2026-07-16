#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:=python3}"
: "${SOURCE_CHECKPOINT:=runs/stage0_convnext_scratch_v2/best.pt}"
: "${RUN_DIR:=runs/stage0_convnext_scratch_v3_160}"
: "${NUM_WORKERS:=4}"
: "${EPOCHS:=20}"

mkdir -p "$RUN_DIR"

"$PYTHON" -m imagenette_stage0.train \
  --arch convnext_tiny \
  --no-pretrained \
  --resume-from "$SOURCE_CHECKPOINT" \
  --dataset-size 160 \
  --image-size 160 \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --num-workers "$NUM_WORKERS" \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --warmup-epochs 1 \
  --label-smoothing 0.05 \
  --mixup-alpha 0 \
  --cutmix-alpha 0 \
  --trivial-augment \
  --random-erasing 0 \
  --early-stopping-patience 8 \
  --seed 42 \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

"$PYTHON" -m imagenette_stage0.evaluate \
  --checkpoint "$RUN_DIR/best.pt" \
  --dataset-size 160 \
  --image-size 160 \
  --batch-size 64 \
  --num-workers "$NUM_WORKERS" \
  --output-csv "$RUN_DIR/eval_clean.csv" \
  2>&1 | tee "$RUN_DIR/eval_clean.log"

"$PYTHON" -m imagenette_stage0.plot_training \
  --jsonl "$RUN_DIR/metrics.jsonl" \
  --title "Stage 0: ConvNeXt Tiny fine-tune at 160" \
  --output "$RUN_DIR/training_curves.png" \
  --output-csv "$RUN_DIR/training_history.csv"
