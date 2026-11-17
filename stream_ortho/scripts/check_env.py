"""S0: environment report (versions, CUDA extensions, checkpoints, data).

Writes ``results/stage-S0/env_report.json`` and a Markdown summary.

Run::

    CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=8.9 \
    python -m stream_ortho.scripts.check_env
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

from stream_ortho import config as zcfg
from stream_ortho import compat


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover
        return f"<failed: {exc}>"


def collect() -> dict:
    rep = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_home_env": os.environ.get("CUDA_HOME"),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvcc_system": _run(["bash", "-lc", "nvcc --version | tail -2 | head -1"]),
        "nvcc_11_8": _run(["/usr/local/cuda-11.8/bin/nvcc", "--version"]).splitlines()[-1]
        if os.path.exists("/usr/local/cuda-11.8/bin/nvcc") else "missing",
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader"]),
        "upstream_commit": compat.upstream_commit(),
    }
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["torch_cuda"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["device_name"] = torch.cuda.get_device_name(0)
            rep["device_capability"] = list(torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info()
            rep["device_total_gb"] = round(total / 1e9, 2)
            rep["device_free_gb"] = round(free / 1e9, 2)
    except Exception as exc:
        rep["torch"] = f"<failed: {exc}>"

    for mod in ("gsplat", "lietorch", "in3d", "mast3r", "cv2", "scipy", "pytest", "evo", "lpips"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "unknown")
            rep[f"{mod}_path"] = getattr(m, "__file__", "")
        except Exception as exc:
            rep[mod] = f"<missing: {exc}>"

    rep["checkpoints"] = zcfg.checkpoints_status()
    rep["paths"] = {
        "PROJ": str(zcfg.PROJ),
        "MAST3R_SLAM_DIR": str(zcfg.MAST3R_SLAM_DIR),
        "NPU_ROOT": str(zcfg.dataset_root()),
    }
    du = shutil.disk_usage(zcfg.PROJ)
    rep["disk_free_gb"] = round(du.free / 1e9, 1)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description="S0: environment report")
    p.add_argument("--out", default=str(zcfg.RESULTS_DIR / "stage-S0"))
    args = p.parse_args(argv)
    out = zcfg.results_dir("stage-S0")
    rep = collect()
    (out / "env_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    md = ["# S0 environment report", ""]
    for k in ("python", "torch", "torch_cuda", "cuda_available", "device_name",
              "device_capability", "gpu", "gsplat", "lietorch", "in3d", "mast3r",
              "cv2", "scipy", "evo", "nvcc_system", "nvcc_11_8", "torch_cuda_arch_list",
              "upstream_commit", "disk_free_gb"):
        md.append(f"- **{k}**: `{rep.get(k)}`")
    md += ["", "## checkpoints", "", "| name | exists | size (MB) | path |", "|---|---|---|---|"]
    for name, info in rep["checkpoints"].items():
        md.append(f"| {name} | {info['exists']} | {info['size_mb']} | `{info['path']}` |")
    (out / "ENV_REPORT.md").write_text("\n".join(md) + "\n")

    print(json.dumps({k: rep[k] for k in ("python", "torch", "torch_cuda", "device_name",
                                          "gsplat", "nvcc_11_8", "disk_free_gb")}, indent=2))
    print(f"[S0] wrote {out/'ENV_REPORT.md'} and env_report.json")


if __name__ == "__main__":
    main()
