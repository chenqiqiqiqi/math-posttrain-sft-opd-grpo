#!/bin/bash
# 剩余校准（协议补充 3 第 3 条），使用速度测试选定的设置 V2：OFFLOAD=1、16384 token/卡、vLLM 0.7、老师批 8。
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
M=/root/autodl-tmp/models
S=runs/calib/rest_status.txt
cleanup() { ray stop --force >/dev/null 2>&1; sleep 3; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done; sleep 5; }
FAST="NGPUS=2 MODEL_DTYPE=fp32 OFFLOAD=1 PPO_MAX_TOKEN_LEN_PER_GPU=16384 GPU_MEM_UTIL=0.7 RM_MICRO=8"
# R1 D-GRPO 1 步
env $FAST ARM=D-GRPO STUDENT=$M/S0 MAX_STEPS=1 SAVE_FREQ=-1 RUN_NAME=calib2-dgrpo bash scripts/train_verl.sh > runs/calib/r1_grpo.log 2>&1; echo "R1_EXIT=$?" >> $S; cleanup
# R2 老师生成 SFT 数据：512 题、上限 7168、双卡数据并行 → 吞吐量与截断率
python scripts/sft_gen.py --teacher $M/T --n-prompts 512 --max-tokens 7168 --gpu-ids 0,1 --out runs/sft_data/calib2 > runs/calib/r2_sftgen.log 2>&1; echo "R2_EXIT=$?" >> $S; cleanup
# R3 D-SFT 3 步（数据不足 192 条时复制补足，仅测速度）
python -c "
import pandas as pd
d = pd.read_parquet('runs/sft_data/calib2/sft_data.parquet')
pd.concat([d] * (192 // max(len(d), 1) + 1)).iloc[:192].to_parquet('runs/sft_data/calib2/sft_data_192.parquet', index=False)
print('kept', len(d))"
SFT_DATA=runs/sft_data/calib2/sft_data_192.parquet STUDENT=$M/S0 NGPUS=2 MODEL_DTYPE=fp32 MICRO=${SFT_MICRO:-1} MAX_STEPS=3 RUN_NAME=calib2-sft \
  bash scripts/train_sft.sh > runs/calib/r3_sft.log 2>&1; echo "R3_EXIT=$?" >> $S; cleanup
# R4 Q-OPD 1 步（显存与 QB 停止行为）
env $FAST ARM=Q-OPD STUDENT=$M/QB TEACHER=$M/QT MAX_STEPS=1 SAVE_FREQ=-1 RUN_NAME=calib2-qopd bash scripts/train_verl.sh > runs/calib/r4_qopd.log 2>&1; echo "R4_EXIT=$?" >> $S; cleanup
echo REST_DONE >> $S
