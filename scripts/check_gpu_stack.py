"""P0 末尾的软件栈自检：确认 torch / flash-attn / vLLM 能在当前 GPU 上真正运行（新架构如 Blackwell sm_120 尤其需要）。
用法：python scripts/check_gpu_stack.py [模型路径，默认 /root/autodl-tmp/models/S0]
"""
import os
import sys

import torch

# 服务器 CUDA 工具包是 13.0 且缺 curand.h，FlashInfer 采样算子无法现场编译；改用 vLLM 的 PyTorch 原生 top-p 采样（分布相同）
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "arch list:", torch.cuda.get_arch_list())

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print("torch bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)

    from flash_attn import flash_attn_func  # noqa: E402

    q = torch.randn(2, 512, 16, 128, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    print("flash-attn ok:", tuple(out.shape))

    from vllm import LLM, SamplingParams  # noqa: E402

    model = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/models/S0"
    llm = LLM(model=model, gpu_memory_utilization=0.5, max_model_len=4096, seed=0)
    text = llm.generate(["1+1="], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text
    print("vLLM ok:", repr(text))
    print("ALL OK")


if __name__ == "__main__":  # vLLM 在 CUDA 已初始化时用 spawn 启动子进程，需要 main 保护
    main()
