#!/bin/bash
# 正式卡型校准（协议补充 1/2）：实测各组单步耗时，用于冻结 B 与 N_SFT。
# 全部用正式设置（fp32、回答上限 7168），只用训练集与 MATH-500，不碰开发集/测试集。
# 用法：NGPUS=2 bash scripts/calibrate.sh   （有卡模式下，项目根目录）
# 显存设置先试不卸载；某项 OOM 时自动以 OFFLOAD=1 重跑该项。
cd "$(dirname "$0")/.."
source /root/pb-venv/bin/activate
M=/root/autodl-tmp/models
S=runs/calib/status.txt
NGPUS=${NGPUS:-2}
GPU_IDS=$(seq -s, 0 $((NGPUS - 1)))
mkdir -p runs/calib
cleanup() { ray stop --force >/dev/null 2>&1; sleep 3; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill "$p" 2>/dev/null; done; sleep 5; }
is_oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$1"; }
mem() { echo "NGPUS=$NGPUS MODEL_DTYPE=fp32 OFFLOAD=$1 PPO_MAX_TOKEN_LEN_PER_GPU=${TOKLEN:-8192} RM_MICRO=${RM_MICRO:-4} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}"; }
# run_rl <tag> <其余 env>：先不卸载，OOM 则卸载重跑
run_rl() {
  local tag=$1; shift
  env $(mem 0) "$@" RUN_NAME=calib-$tag bash scripts/train_verl.sh > runs/calib/$tag.log 2>&1; local rc=$?; cleanup
  if [ $rc -ne 0 ] && is_oom runs/calib/$tag.log; then
    echo "${tag}_OOM_RETRY_OFFLOAD" >> $S
    rm -rf runs/train/calib-$tag
    env $(mem 1) "$@" RUN_NAME=calib-$tag bash scripts/train_verl.sh > runs/calib/$tag.log 2>&1; rc=$?; cleanup
  fi
  echo "${tag}_EXIT=$rc" >> $S
}

# C1 D-OPD 2 步（不存检查点：数据盘空间不足；C6 复用功能测试 F1 留下的检查点）
run_rl dopd ARM=D-OPD STUDENT=$M/S0 TEACHER=$M/T MAX_STEPS=2 SAVE_FREQ=-1
# C2 D-GRPO 1 步
run_rl dgrpo ARM=D-GRPO STUDENT=$M/S0 MAX_STEPS=1 SAVE_FREQ=-1
# C3 老师生成 SFT 数据：512 题、上限 7168、多卡数据并行 → 吞吐量与截断率
python scripts/sft_gen.py --teacher $M/T --n-prompts 512 --max-tokens 7168 --gpu-ids $GPU_IDS --out runs/sft_data/calib > runs/calib/sftgen.log 2>&1
echo "sftgen_EXIT=$?" >> $S; cleanup
# C4 SFT 吞吐量：3 步（数据不足 192 条时复制补足，仅测速度）
python - <<'PY'
import pandas as pd
d = pd.read_parquet("runs/sft_data/calib/sft_data.parquet")
pd.concat([d] * (192 // max(len(d), 1) + 1)).iloc[:192].to_parquet("runs/sft_data/calib/sft_data_192.parquet", index=False)
print("kept", len(d))
PY
SFT_DATA=runs/sft_data/calib/sft_data_192.parquet STUDENT=$M/S0 NGPUS=$NGPUS MODEL_DTYPE=fp32 MICRO=${SFT_MICRO:-1} MAX_STEPS=3 RUN_NAME=calib-sft \
  bash scripts/train_sft.sh > runs/calib/sft.log 2>&1; echo "sft_EXIT=$?" >> $S; cleanup
# C5 Q-OPD 1 步（检查显存与 QB 的停止行为）
run_rl qopd ARM=Q-OPD STUDENT=$M/QB TEACHER=$M/QT MAX_STEPS=1 SAVE_FREQ=-1
# C6 检查点合并为 HF 格式 + vLLM（bf16）在 MATH-500 前 50 题上生成（只验证流程）
CK=${MERGE_CK:-runs/train/func-grpo/ckpt/global_step_1/actor}
python -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir runs/calib/merged-dopd2 > runs/calib/merge.log 2>&1
python -c "import pandas as pd; pd.read_parquet('data/prepared/sanity_math500.parquet').iloc[:50].to_parquet('runs/calib/math500_50.parquet', index=False)"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_generate.py --model runs/calib/merged-dopd2 --name merged --data runs/calib/math500_50.parquet --k 1 --max-tokens 8192 --out runs/calib/merged_eval >> runs/calib/merge.log 2>&1
python scripts/eval_grade.py runs/calib/merged_eval >> runs/calib/merge.log 2>&1; echo "merge_EXIT=$?" >> $S; cleanup
# C7 推迟的 P1：T 在 MATH-500 上 k=4（单卡，同时得到每卡评测吞吐量）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_generate.py --model $M/T --name T --data data/prepared/sanity_math500.parquet --k 4 --max-tokens 31744 --out runs/calib/math500/T > runs/calib/t_math500.log 2>&1
python scripts/eval_grade.py runs/calib/math500/T >> runs/calib/t_math500.log 2>&1; echo "t_math500_EXIT=$?" >> $S; cleanup
echo CALIB_DONE >> $S
