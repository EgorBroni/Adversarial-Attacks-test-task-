#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:=python3}"
: "${RUN_DIR:=runs/stage0_convnext_scratch}"
: "${NUM_WORKERS:=4}"
: "${EPOCHS:=30}"

mkdir -p "$RUN_DIR"

"$PYTHON" -m imagenette_stage0.train \
  --arch convnext_tiny \
  --no-pretrained \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --image-size 160 \
  --num-workers "$NUM_WORKERS" \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

"$PYTHON" -m imagenette_stage0.evaluate \
  --checkpoint "$RUN_DIR/best.pt" \
  --num-workers "$NUM_WORKERS" \
  --output-csv "$RUN_DIR/eval_clean.csv" \
  2>&1 | tee "$RUN_DIR/eval_clean.log"

"$PYTHON" -m imagenette_stage0.plot_training \
  --jsonl "$RUN_DIR/metrics.jsonl" \
  --title "Stage 0: randomly initialized ConvNeXt Tiny" \
  --output "$RUN_DIR/training_curves.png" \
  --output-csv "$RUN_DIR/training_history.csv"
