#!/bin/bash
# The study's setup: AyroX hybrid drone in the NUST H-12 and UET Taxila twins.
# Needs assets/ from tools/build_twins.py + tools/build_maps.py. One Isaac Sim process at a time;
# dataset generation is resumable, evaluation re-runs its split.
cd "$(dirname "$0")"
ISAAC=~/Documents/isaacsim/python.sh
mkdir -p logs
for spec in nust_train_clean:${CLEAN:-50} nust_train_varied:${VARIED:-200}; do
  split=${spec%%:*}; n=${spec##*:}
  $ISAAC generate_dataset.py --split $split --flights $n 2>&1 | tee logs/gen_$split.log | grep --line-buffered -E "^\[nust|^done"
done
for split in nust_test_new_layouts taxila_test_unseen nust_test_harsh; do
  $ISAAC evaluate.py --split $split --flights "${EVAL:-50}" 2>&1 | tee logs/eval_$split.log | grep --line-buffered -E "^\[(nust|taxila)|success_rate"
done
