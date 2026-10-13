"""用 vLLM 为一个模型在若干数据集上生成评测回答（协议第 4 节）。

每题采样 k 次（SamplingParams.n=k），全局随机种子固定。输出 JSONL，每行一条回答：
  id, set, sample_idx, response, num_tokens, finish_reason
以及同名 .meta.json 记录模型、解码参数、版本、耗时，便于审计和计费。

示例：
  python scripts/eval_generate.py --model /root/models/S0 --name S0 \
      --data data/prepared/test_aime24.parquet data/prepared/test_aime25.parquet \
      --k 64 --max-tokens 31744 --out runs/eval/test/S0
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

# 服务器 CUDA 工具包是 13.0 且缺 curand.h，FlashInfer 采样算子无法现场编译；改用 vLLM 的 PyTorch 原生 top-p 采样（分布相同）
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="模型代号，如 S0 / T / D-OPD")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-thinking", choices=["true", "false", "none"], default="none",
                    help="Qwen3 族传 false；R1-Distill 的模板不认这个参数，用 none")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16", help="所有模型统一 bf16 推理（fp32 训练出的检查点也按 bf16 评测）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import vllm
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, seed=args.seed, tensor_parallel_size=args.tp, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_tokens + 2048)
    tok = llm.get_tokenizer()
    tmpl_kwargs = {} if args.enable_thinking == "none" else {"enable_thinking": args.enable_thinking == "true"}
    # 显式 top_k=-1：vLLM 默认会套用模型自带 generation_config（如 Qwen3-4B 的 top_k=20），不同模型解码会不一致。
    # 停止符同上游 gen_vllm.py：Qwen3-Base 的 eos 只有 <|endoftext|>，需另加 <|im_end|>。
    stop_ids = [i for t in ['<|im_end|>', '<|endoftext|>'] if (i := tok.convert_tokens_to_ids(t)) is not None
                and i != tok.unk_token_id]
    params = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                            max_tokens=args.max_tokens, stop_token_ids=stop_ids or None)

    for path in args.data:
        set_name = Path(path).stem
        out_file = out_dir / f"{set_name}.jsonl"
        if out_file.exists():
            print(f"已存在，跳过：{out_file}")
            continue
        df = pd.read_parquet(path)
        prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True, **tmpl_kwargs)
                   for p in df["prompt"]]
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = time.time() - t0
        n_tokens = 0
        with out_file.open("w", encoding="utf-8") as f:
            for (_, row), out in zip(df.iterrows(), outputs):
                for j, o in enumerate(out.outputs):
                    n_tokens += len(o.token_ids)
                    f.write(json.dumps({
                        "id": row["extra_info"]["id"], "set": set_name, "sample_idx": j,
                        "response": o.text, "num_tokens": len(o.token_ids), "finish_reason": o.finish_reason,
                    }, ensure_ascii=False) + "\n")
        meta = {
            "name": args.name, "model": args.model, "data": path, "rows": len(df), "k": args.k,
            "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
            "seed": args.seed, "enable_thinking": args.enable_thinking, "tp": args.tp, "top_k": -1, "stop_token_ids": stop_ids, "dtype": args.dtype,
            "vllm": vllm.__version__, "elapsed_sec": round(elapsed, 1), "generated_tokens": n_tokens,
            "example_prompt": prompts[0],
        }
        (out_dir / f"{set_name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{set_name}: {len(df)} 题 × {args.k}，{n_tokens} token，{elapsed:.0f}s")


if __name__ == "__main__":
    main()
