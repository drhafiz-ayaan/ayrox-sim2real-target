#!/bin/bash
# Run one split as several single-drone Isaac Sim processes in parallel, each on its own flight range.
#   bash run_sharded.sh <split> <first> <count> [shards] [flights_per_process]
# Same code and seeds as a single run, so the dataset is identical, only faster.
# Limits found on a 31 GB / RTX 3080 Laptop machine:
#  - each process uses ~5-6 GB RAM (one grew to 10.7 GB): keep shards x 6 GB well under free memory
#  - after ~85 resets in one process RTX dies with "Out of resource descriptors" and the process
#    hangs, so every shard restarts Isaac each `flights_per_process` flights (default 40)
cd "$(dirname "$0")"
split=$1; first=$2; count=$3; shards=${4:-2}; chunk=${5:-40}
per=$(( (count + shards - 1) / shards ))
mkdir -p logs
for ((k = 0; k < shards; k++)); do
  start=$(( first + k * per ))
  end=$(( start + per )); (( end > first + count )) && end=$(( first + count ))
  (( end > start )) || continue
  (
    for ((s = start; s < end; s += chunk)); do
      n=$(( end - s )); (( n > chunk )) && n=$chunk
      ~/Documents/isaacsim/python.sh generate_dataset.py --split "$split" --first $s --flights $n >> "logs/gen_${split}_$start.log" 2>&1
    done
  ) &
done
wait
grep -hE "^\[|^done|WARNING" logs/gen_${split}_*.log | grep -vE "^\[[0-9.]+s\]|absl" | tail -n 5
echo "all shards finished"
