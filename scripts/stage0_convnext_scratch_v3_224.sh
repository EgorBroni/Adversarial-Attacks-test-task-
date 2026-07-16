#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:=python3}"
: "${SOURCE_CHECKPOINT:=runs/stage0_convnext_scratch_v2/best.pt}"
: "${RUN_DIR:=runs/stage0_convnext_scratch_v3_224}"
: "${NUM_WORKERS:=4}"
: "${EPOCHS:=25}"

mkdir -p "$RUN_DIR"

"$PYTHON" -m imagenette_stage0.evaluate \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --dataset-size 320 \
  --image-size 224 \
  --batch-size 32 \
  --num-workers "$NUM_WORKERS" \
  --output-csv "$RUN_DIR/source_eval_clean.csv" \
  2>&1 | tee "$RUN_DIR/source_eval_clean.log"

"$PYTHON" -m imagenette_stage0.train \
  --arch convnext_tiny \
  --no-pretrained \
  --resume-from "$SOURCE_CHECKPOINT" \
  --dataset-size 320 \
  --image-size 224 \
  --epochs "$EPOCHS" \
  --batch-size 16 \
  --num-workers "$NUM_WORKERS" \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --warmup-epochs 2 \
  --label-smoothing 0.05 \
  --mixup-alpha 0.1 \
  --cutmix-alpha 0 \
  --trivial-augment \
  --random-erasing 0 \
  --early-stopping-patience 10 \
  --seed 42 \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

"$PYTHON" -m imagenette_stage0.evaluate \
  --checkpoint "$RUN_DIR/best.pt" \
  --dataset-size 320 \
  --image-size 224 \
  --batch-size 32 \
  --num-workers "$NUM_WORKERS" \
  --output-csv "$RUN_DIR/eval_clean.csv" \
  2>&1 | tee "$RUN_DIR/eval_clean.log"

"$PYTHON" -m imagenette_stage0.plot_training \
  --jsonl "$RUN_DIR/metrics.jsonl" \
  --title "Stage 0: ConvNeXt Tiny progressive resize 224" \
  --output "$RUN_DIR/training_curves.png" \
  --output-csv "$RUN_DIR/training_history.csv"
