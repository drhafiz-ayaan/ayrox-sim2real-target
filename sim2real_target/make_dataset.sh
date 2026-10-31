#!/bin/bash
# Generate the training datasets headless, one Isaac Sim process at a time.
# Resumable: re-running skips flights that already exist in data/<split>/.
cd "$(dirname "$0")"
ISAAC=~/Documents/isaacsim/python.sh
mkdir -p logs
$ISAAC generate_dataset.py --split train_clean --flights "${CLEAN:-50}" 2>&1 | tee logs/gen_train_clean.log | grep --line-buffered -E "^\[train|^done"
$ISAAC generate_dataset.py --split train_varied --flights "${VARIED:-300}" 2>&1 | tee logs/gen_train_varied.log | grep --line-buffered -E "^\[train|^done"
# Expert baseline on the held-out test splits.
for split in test_new_layouts test_unseen_campus test_harsh; do
  $ISAAC evaluate.py --split $split --flights "${EVAL:-25}" 2>&1 | tee logs/eval_$split.log | grep --line-buffered -E "^\[test|success_rate"
done
