"""试跑 P3：OPD 逐 token 奖励在 fp32 与 bf16 下是否一致（协议第 7 节）。

做法（与上游 dp_actor 中 only_stu + student_p 的公式相同）：
  1. 学生 S0 用 vLLM 在训练集前 8 题上各生成 1 条回答（温度 1.0，最长 7168）；
  2. 分别在 fp32 和 bf16 下，用 HF transformers 前向学生与老师，得到每个位置：
       学生 top-16 的 token 与 log p；老师在这 16 个 token 上的 log q；
       权重 w = softmax(学生 top-16 的 log p)；逐位置奖励 r_t = Σ_k w_k (log q_k − log p_k)
  3. 比较两种精度的 r_t：Pearson 相关、归一化误差 median|Δr| / median|r_fp32|、符号一致率，
     以及 top-16 集合的一致率。判定规则见协议 P3。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 服务器 CUDA 工具包是 13.0 且缺 curand.h，FlashInfer 采样算子无法现场编译；改用 vLLM 的 PyTorch 原生 top-p 采样（分布相同）
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[1]
K = 16


@torch.no_grad()
def token_rewards(student, teacher, input_ids, prompt_len):
    s = torch.log_softmax(student(input_ids).logits[0, prompt_len - 1:-1].float(), -1)
    t = torch.log_softmax(teacher(input_ids).logits[0, prompt_len - 1:-1].float(), -1)
    s_logp, ids = s.topk(K, dim=-1)
    t_logp = t.gather(-1, ids)
    w = torch.softmax(s_logp, -1)
    return (w * (t_logp - s_logp)).sum(-1).cpu().numpy(), ids.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "runs" / "pilot" / "precision_check.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(ROOT / "data" / "prepared" / "train_dapo.parquet").iloc[: args.n]
    tok = AutoTokenizer.from_pretrained(args.student)
    prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True) for p in df["prompt"]]
    llm = LLM(model=args.student, seed=42, gpu_memory_utilization=0.5, max_model_len=8192)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=7168))
    seqs = [(tok(p, add_special_tokens=False)["input_ids"], list(o.outputs[0].token_ids)) for p, o in zip(prompts, outs)]
    del llm
    torch.cuda.empty_cache()

    res = {}
    for dtype in [torch.float32, torch.bfloat16]:
        stu = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
        tea = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
        res[str(dtype)] = [token_rewards(stu, tea, torch.tensor([p + r]).cuda(), len(p)) for p, r in seqs]
        del stu, tea
        torch.cuda.empty_cache()

    r32 = np.concatenate([r for r, _ in res["torch.float32"]])
    r16 = np.concatenate([r for r, _ in res["torch.bfloat16"]])
    id32 = np.concatenate([i for _, i in res["torch.float32"]])
    id16 = np.concatenate([i for _, i in res["torch.bfloat16"]])
    topk_agree = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(id32, id16)]))
    report = {
        "positions": int(len(r32)), "response_lengths": [len(r) for _, r in seqs],
        "pearson_r": float(np.corrcoef(r32, r16)[0, 1]),
        "norm_err": float(np.median(np.abs(r16 - r32)) / np.median(np.abs(r32))),
        "sign_agree": float(np.mean(np.sign(r32) == np.sign(r16))),
        "topk_set_agree": topk_agree,
        "mean_r_fp32": float(r32.mean()), "mean_r_bf16": float(r16.mean()),
    }
    report["decision"] = "bf16" if report["pearson_r"] >= 0.99 and report["norm_err"] <= 0.05 else "fp32"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
