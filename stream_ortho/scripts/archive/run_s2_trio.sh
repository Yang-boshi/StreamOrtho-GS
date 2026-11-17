#!/usr/bin/env bash
# Week 2: MASt3R-SLAM baseline on the three sequences kept in PROJ/data.
set -u
cd /home/zxy/syb/ZSO/PROJ/code
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/home/zxy/anaconda3/envs/2dgs-slam/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9
PY=/home/zxy/anaconda3/envs/2dgs-slam/bin/python
PROJ=/home/zxy/syb/ZSO/PROJ
DATA=$PROJ/data
LOG=$PROJ/results/week2/logs
mkdir -p "$LOG"

for seq in phantom3-factory-kfs phantom3-ieu-kfs phantom3-npu-kfs; do
  tag="${seq#phantom3-}"; tag="${tag%-kfs}"
  echo "===== $tag $(date +%H:%M:%S) ====="
  $PY -m stream_ortho.scripts.run_mast3r_baseline \
      --dataset "$DATA/$seq" --out "$PROJ/results/stage-S2/${tag}_calib" --calib \
      > "$LOG/s2_${tag}_calib.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)"
done
echo "ALL DONE $(date +%H:%M:%S)"
