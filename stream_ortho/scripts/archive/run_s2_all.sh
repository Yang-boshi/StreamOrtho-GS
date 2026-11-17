#!/usr/bin/env bash
# S2: run the MASt3R-SLAM baseline on the three week-1 sequences, with/without calib.
set -u
cd /home/zxy/syb/ZSO/PROJ/code

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:/home/zxy/anaconda3/envs/2dgs-slam/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9
PY=/home/zxy/anaconda3/envs/2dgs-slam/bin/python
PROJ=/home/zxy/syb/ZSO/PROJ
NPU=$PROJ/data/NPU
LOG=$PROJ/results/week1/logs

run () {  # $1=seq dir  $2=tag  $3=extra args
  local seq="$1" tag="$2"; shift 2
  echo "===== $tag  $(date +%H:%M:%S) ====="
  $PY -m stream_ortho.scripts.run_mast3r_baseline \
      --dataset "$NPU/$seq" \
      --out "$PROJ/results/stage-S2/$tag" \
      "$@" > "$LOG/s2_$tag.log" 2>&1
  echo "exit=$? $(date +%H:%M:%S)  $(tail -2 "$LOG/s2_$tag.log" | head -1)"
}

run gopro-monticules-kfs monticules_nocalib
run gopro-monticules-kfs monticules_calib   --calib
run phantom3-npu-kfs    npu_nocalib
run phantom3-npu-kfs    npu_calib          --calib
run phantom3-centralPark-kfs centralPark_nocalib
run phantom3-centralPark-kfs centralPark_calib --calib

echo "ALL DONE $(date +%H:%M:%S)"
