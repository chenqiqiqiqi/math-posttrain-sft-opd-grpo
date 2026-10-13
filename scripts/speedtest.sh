#!/bin/bash
# 显存/速度设置对比（只影响速度，不改变算法）：D-OPD 各跑 1 步，与基准 OFFLOAD=0/8192/0.4（515–520 秒）比较。
# V2 同时测试保存检查点与 ckpt_exporter（bf16 HF 导出）。
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
M=/root/autodl-tmp/models
S=runs/calib/speed_status.txt
cleanup() { ray stop --force >/dev/null 2>&1; sleep 3; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done; sleep 5; }
BASE="ARM=D-OPD STUDENT=$M/S0 TEACHER=$M/T NGPUS=2 MODEL_DTYPE=fp32 RM_MICRO=8 MAX_STEPS=1"
env $BASE OFFLOAD=0 PPO_MAX_TOKEN_LEN_PER_GPU=16384 GPU_MEM_UTIL=0.5 SAVE_FREQ=-1 RUN_NAME=speed-v1 \
  bash scripts/train_verl.sh > runs/calib/speed_v1.log 2>&1; echo "V1_EXIT=$?" >> $S; cleanup
env $BASE OFFLOAD=1 PPO_MAX_TOKEN_LEN_PER_GPU=16384 GPU_MEM_UTIL=0.7 SAVE_FREQ=1 MAX_CKPT_KEEP=1 RUN_NAME=speed-v2 \
  bash scripts/train_verl.sh > runs/calib/speed_v2.log 2>&1; echo "V2_EXIT=$?" >> $S; cleanup
CUDA_VISIBLE_DEVICES= python scripts/ckpt_exporter.py --run-dir runs/train/speed-v2 --once > runs/calib/exporter.log 2>&1; echo "EXPORT_EXIT=$?" >> $S
CUDA_VISIBLE_DEVICES=0 python scripts/eval_generate.py --model runs/train/speed-v2/hf/step_1 --name v2 --data runs/calib/math500_50.parquet --k 1 --max-tokens 8192 --out runs/calib/v2_eval > runs/calib/v2_eval.log 2>&1
python scripts/eval_grade.py runs/calib/v2_eval >> runs/calib/v2_eval.log 2>&1; echo "V2EVAL_EXIT=$?" >> $S; cleanup
echo SPEED_DONE >> $S
