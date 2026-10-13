"""D-SFT 第一步：老师在训练题上生成回答，作为 SFT 数据（协议 3.1）。

- 按 train_dapo.parquet 的固定题序取前 N 题，每题 m 条；温度 0.7、top_p 0.95（论文附录 C.1）；
- 只过滤被截断的回答（finish_reason == length），不按对错过滤；
- response 是老师在"生成提示"之后输出的原文（R1 模板的生成提示以 <think>\\n 结尾，思考内容完整保留）；
- 多卡时数据并行（--gpu-ids 0,1）：每张卡一个 vLLM 进程，按题号交错分片，合并时恢复原题序。
  两张卡都在干活，D-SFT 的 GPU 小时 = 卡数 × 墙钟，与其他组口径一致；
- 墙钟耗时写入 meta，计入 D-SFT 的 GPU 小时预算。

示例：python scripts/sft_gen.py --teacher /root/autodl-tmp/models/T --n-prompts 500 --gpu-ids 0,1 --out runs/sft_data/pilot
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# 服务器 CUDA 工具包是 13.0 且缺 curand.h，FlashInfer 采样算子无法现场编译；改用 vLLM 的 PyTorch 原生 top-p 采样（分布相同）
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[1]


def run_shard(args):
    """单卡工作进程：处理第 shard 片（题号 ≡ shard mod num_shards）。
    分块生成（每块 CHUNK 题），每块完成即落盘 shard{i}_part{j}.parquet；重跑时跳过已完成的块。
    每块的生成耗时记入 shard{i}_parts.jsonl。"""
    import json as _json
    import time as _time
    from vllm import LLM, SamplingParams

    CHUNK = 2000
    out = Path(args.out)
    df = pd.read_parquet(args.data).iloc[: args.n_prompts]
    df = df.iloc[args.shard::args.num_shards]
    chunks = [df.iloc[i:i + CHUNK] for i in range(0, len(df), CHUNK)]
    todo = [j for j in range(len(chunks)) if not (out / f"shard{args.shard}_part{j}.parquet").exists()]
    if not todo:
        return
    llm = LLM(model=args.teacher, seed=args.seed + args.shard, gpu_memory_utilization=0.9,
              max_model_len=1024 + args.max_tokens)
    tok = llm.get_tokenizer()
    params = SamplingParams(n=args.samples_per_prompt, temperature=0.7, top_p=0.95, top_k=-1, max_tokens=args.max_tokens)
    for j in todo:
        part = chunks[j]
        prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True) for p in part["prompt"]]
        t0 = _time.time()
        outputs = llm.generate(prompts, params)
        rows = []
        for (pos, r), o in zip(part.iterrows(), outputs):
            for k, c in enumerate(o.outputs):
                rows.append({"order": pos * 1000 + k, "id": r["extra_info"]["id"], "prompt": r["prompt"][0]["content"],
                             "response": c.text, "num_tokens": len(c.token_ids), "finish_reason": c.finish_reason})
        pd.DataFrame(rows).to_parquet(out / f"shard{args.shard}_part{j}.parquet", index=False)
        with open(out / f"shard{args.shard}_parts.jsonl", "a") as f:
            f.write(_json.dumps({"part": j, "prompts": len(part), "gen_sec": round(_time.time() - t0, 1)}) + "\n")
        if j == 0:
            (out / f"shard{args.shard}_prompt.txt").write_text(prompts[0], encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "prepared" / "train_dapo.parquet"))
    ap.add_argument("--n-prompts", type=int, required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=7168)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-ids", default="0", help="数据并行用的卡，如 0,1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.shard >= 0:
        return run_shard(args)

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = args.gpu_ids.split(",")
    procs = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, __file__, "--teacher", args.teacher, "--data", args.data,
               "--n-prompts", str(args.n_prompts), "--samples-per-prompt", str(args.samples_per_prompt),
               "--max-tokens", str(args.max_tokens), "--seed", str(args.seed), "--out", args.out,
               "--shard", str(i), "--num-shards", str(len(gpus))]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"shard{i}.log", "w"), stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    wall = time.time() - t_start
    with open(out / "gen_attempts.jsonl", "a") as f:
        f.write(json.dumps({"wall_sec": round(wall, 1), "codes": codes}) + "\n")
    wall_total = sum(json.loads(l)["wall_sec"] for l in open(out / "gen_attempts.jsonl"))
    if any(codes):
        raise SystemExit(f"分片进程失败：{codes}，见 {out}/shard*.log")

    allrows = pd.concat([pd.read_parquet(f) for i in range(len(gpus)) for f in sorted(out.glob(f"shard{i}_part*.parquet"))])
    allrows = allrows.sort_values("order").reset_index(drop=True)
    kept = allrows[allrows["finish_reason"] != "length"]
    kept[["id", "prompt", "response", "num_tokens"]].to_parquet(out / "sft_data.parquet", index=False)
    meta = {
        "teacher": args.teacher, "n_prompts": args.n_prompts, "samples_per_prompt": args.samples_per_prompt,
        "generated": len(allrows), "kept": len(kept), "truncated_dropped": int((allrows["finish_reason"] == "length").sum()),
        "truncation_rate": round(float((allrows["finish_reason"] == "length").mean()), 4),
        "generated_tokens": int(allrows["num_tokens"].sum()), "kept_mean_tokens": round(float(kept["num_tokens"].mean()), 1) if len(kept) else 0,
        "wall_sec_including_load": round(wall_total, 1), "wall_sec_last_attempt": round(wall, 1),
        "gpus": gpus, "gpu_sec": round(wall_total * len(gpus), 1),
        "seed": args.seed, "example_prompt": (out / "shard0_prompt.txt").read_text(encoding="utf-8"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "example_prompt"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
