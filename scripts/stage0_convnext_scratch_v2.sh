#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:=python3}"
: "${RUN_DIR:=runs/stage0_convnext_scratch_v2}"
: "${NUM_WORKERS:=4}"
: "${EPOCHS:=80}"

mkdir -p "$RUN_DIR"

"$PYTHON" -m imagenette_stage0.train \
  --arch convnext_tiny \
  --no-pretrained \
  --epochs "$EPOCHS" \
  --batch-size 32 \
  --image-size 160 \
  --num-workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --trivial-augment \
  --random-erasing 0.1 \
  --early-stopping-patience 15 \
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
  --title "Stage 0: ConvNeXt Tiny scratch recipe v2" \
  --output "$RUN_DIR/training_curves.png" \
  --output-csv "$RUN_DIR/training_history.csv"
