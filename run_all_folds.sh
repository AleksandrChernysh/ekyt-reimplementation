#!/usr/bin/env bash
# Train and evaluate all four folds, then print the summary table.
# Assumes GazeBase is unpacked into data_raw/ and preprocess.py has been run.
set -euo pipefail

for fold in 0 1 2 3; do
  echo "=== Fold ${fold} ==="
  python split.py --fold "${fold}"
  python normalize.py
  python train.py --fold "${fold}"
  python evaluate.py --fold "${fold}"
  python plots.py --fold "${fold}"
done

python summarize.py
