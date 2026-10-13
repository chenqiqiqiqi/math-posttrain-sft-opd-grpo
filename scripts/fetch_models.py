"""在服务器上下载 5 个模型（固定 HuggingFace revision），并逐文件校验（试跑 P0）。

为什么不用 snapshot_download：AutoDL 上直连/学术加速 huggingface.co 很慢或报错（Xet 协议 401），
hf-mirror 单连接约 0.9 MB/s，aria2c 16 连接约 8 MB/s。
文件清单与 sha256 来自 manifests/hf_model_files.json（本地直连 HuggingFace API 生成）。
大文件（LFS）校验 sha256；小文件校验 git blob sha1 不方便，只校验大小。

用法：python scripts/fetch_models.py [--models S0 T QB QS QT] [--mirror https://hf-mirror.com]
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["S0", "T", "QB", "QS", "QT"])
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models")
    ap.add_argument("--mirror", default="https://hf-mirror.com")
    args = ap.parse_args()

    spec = json.loads((ROOT / "manifests" / "hf_model_files.json").read_text())
    lines = []
    for code in args.models:
        m = spec[code]
        for path, meta in m["files"].items():
            dest = Path(args.model_dir) / code / path
            if dest.exists() and dest.stat().st_size == meta["size"] and not Path(str(dest) + ".aria2").exists():
                continue
            lines += [f"{args.mirror}/{m['repo']}/resolve/{m['revision']}/{path}",
                      f"  dir={dest.parent}", f"  out={dest.name}"]
    if lines:
        inp = Path("/tmp/aria2_models.txt")
        inp.write_text("\n".join(lines) + "\n")
        subprocess.run(["aria2c", "-i", str(inp), "-j", "3", "-x", "16", "-s", "16", "-k", "1M",
                        "--file-allocation=none", "--continue=true", "--max-tries=10", "--retry-wait=5",
                        "--summary-interval=60", "--console-log-level=warn"], check=True)

    record, bad = {}, []
    for code in args.models:
        m = spec[code]
        for path, meta in m["files"].items():
            dest = Path(args.model_dir) / code / path
            if not dest.exists() or dest.stat().st_size != meta["size"]:
                bad.append(f"{code}/{path}: size")
            elif meta["sha256"] and sha256(dest) != meta["sha256"]:
                bad.append(f"{code}/{path}: sha256")
        record[code] = {"repo": m["repo"], "revision": m["revision"], "path": f"{args.model_dir}/{code}",
                        "verified": not any(b.startswith(code + "/") for b in bad)}
        print(code, m["repo"], m["revision"][:10], "OK" if record[code]["verified"] else "FAILED", flush=True)
    out = ROOT / "manifests" / "models.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(record)
    out.write_text(json.dumps(old, indent=2))
    if bad:
        raise SystemExit("校验失败：\n" + "\n".join(bad))
    print("ALL MODELS VERIFIED")


if __name__ == "__main__":
    main()
