"""检查点导出器（协议补充 2 第 6 条的保留规则）。

训练进行时在 CPU 上后台运行：每当 verl 写完一个新检查点（以 latest_checkpointed_iteration.txt 为准），
把 ckpt/global_step_N/actor 合并为 HuggingFace 格式并转成 bf16，存到 <run>/hf/step_N；
每组只滚动保留最近 KEEP 个导出（终点 + 终点前 2 个保存点），更早的删除。
续训检查点本身由 verl 的 max_actor_ckpt_to_keep=1 管理。

用法：CUDA_VISIBLE_DEVICES= python scripts/ckpt_exporter.py --run-dir runs/train/D-OPD [--keep 3] [--once]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def ckpt_ready(run_dir: Path, step: int) -> bool:
    g = run_dir / "ckpt" / f"global_step_{step}"
    return (g / "actor").exists() or any((g / "huggingface").glob("*.safetensors"))


def export(run_dir: Path, step: int):
    g = run_dir / "ckpt" / f"global_step_{step}"
    tmp = run_dir / "hf" / f".tmp_step_{step}"
    dst = run_dir / "hf" / f"step_{step}"
    if dst.exists():
        return
    shutil.rmtree(tmp, ignore_errors=True)
    if (g / "actor").exists():  # verl RL 检查点：FSDP 分片 → 合并到临时目录
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(g / "actor"), "--target_dir", str(tmp)], check=True)
        src = tmp
    else:  # verl SFT 检查点：已含 HF 权重（fp32），直接读取，不复制（省磁盘）
        src = g / "huggingface"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    for f in ["generation_config.json", "chat_template.jinja"]:
        if (src / f).exists() and not (dst / f).exists():
            shutil.copy(src / f, dst / f)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[exporter] exported step {step} -> {dst}", flush=True)


def prune(run_dir: Path, keep: int):
    steps = sorted(int(p.name.split("_")[1]) for p in (run_dir / "hf").glob("step_*"))
    for s in steps[:-keep]:
        shutil.rmtree(run_dir / "hf" / f"step_{s}", ignore_errors=True)
        print(f"[exporter] pruned step {s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true", help="只处理当前最新检查点后退出")
    ap.add_argument("--stop-file", default=None, help="该文件出现且已处理完最新检查点时退出")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "hf").mkdir(parents=True, exist_ok=True)
    tracker = run_dir / "ckpt" / "latest_checkpointed_iteration.txt"
    done = set()
    while True:
        if tracker.exists():
            step = int(tracker.read_text().strip())
            if step not in done and ckpt_ready(run_dir, step):
                export(run_dir, step)
                prune(run_dir, args.keep)
                done.add(step)
        if args.once or (args.stop_file and Path(args.stop_file).exists() and
                         (not tracker.exists() or int(tracker.read_text().strip()) in done)):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
