#!/bin/bash
# 服务器环境（试跑 P0）。分阶段，前两步可在 AutoDL 无卡模式下完成以省钱：
#   bash scripts/setup_server.sh env      # 建 venv、装依赖（版本同上游 install_vllm_sglang_mcore.sh，不装 sglang/megatron）
#   bash scripts/setup_server.sh models   # 下载 5 个模型（固定 revision，写 manifests/models.json）
#   bash scripts/setup_server.sh check    # 有卡后：实测 torch / flash-attn / vLLM 能在这张卡上运行
set -euo pipefail
set -x
PB_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/root/.local/bin/python3.12}
VENV=${VENV:-/root/pb-venv}                       # 放系统盘，数据盘留给模型和检查点
MODEL_DIR=${MODEL_DIR:-/root/autodl-tmp/models}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}
STAGE=${1:?用法: setup_server.sh env|models|check}

turbo_on()  { [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1 || true; }
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true; }

case "$STAGE" in
env)
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install -U pip wheel setuptools
  export PIP_CONSTRAINT="$PB_ROOT/scripts/constraints-2025q4.txt"
  # 与上游 verl/scripts/install_vllm_sglang_mcore.sh 相同的版本
  pip install "vllm==0.11.0"
  pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
      "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
      "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
      pytest py-spy tensorboard
  pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
  WHL=flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
  if [ ! -f "$PIP_CACHE_DIR/$WHL" ]; then
    mkdir -p "$PIP_CACHE_DIR"; turbo_on
    curl -L --retry 3 -o "$PIP_CACHE_DIR/$WHL" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/$WHL"
    turbo_off
  fi
  pip install --no-cache-dir "$PIP_CACHE_DIR/$WHL"
  pip install --no-cache-dir flashinfer-python==0.3.1
  pip install --no-deps -e "$PB_ROOT/upstream/Rethinking-OPD/verl"
  pip install math-verify latex2sympy2_extended scipy   # 去重在本地做，服务器不装 sentence-transformers（它会拉 transformers 5）
  python -c "import torch, vllm, verl, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__, 'flash_attn', flash_attn.__version__)"
  pip freeze > "$PB_ROOT/manifests/pip_freeze_server.txt"
  ;;
models)
  # 见 scripts/fetch_models.py：hf-mirror + aria2c 16 连接，固定 revision，逐文件校验 sha256
  source "$VENV/bin/activate"
  which aria2c >/dev/null || apt-get install -y aria2
  python "$PB_ROOT/scripts/fetch_models.py" --model-dir "$MODEL_DIR"
  du -sh "$MODEL_DIR"/*
  ;;
check)
  source "$VENV/bin/activate"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # vLLM 0.11 / torch 2.8 (cu128) 需要驱动 >= 570
  # 失败多半是 flash-attn 预编译包不含当前架构，需源码编译（见 PILOT_RUNBOOK）
  python "$PB_ROOT/scripts/check_gpu_stack.py" "$MODEL_DIR/S0"
  ;;
*) echo "未知阶段 $STAGE"; exit 1 ;;
esac
