"""Environment probe: is this box (and this venv) able to run the benchmark?"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

from .common import free_disk_gib, gpu_inventory, run_cmd, which


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _vllm_patch_status() -> dict:
    """Check that the WeMM patch registered its model classes."""
    status = {"vllm_path": None, "wemm_registered": False, "mtp_registered": False}
    try:
        import vllm  # noqa: F401
        import torch

        package_dir = Path(vllm.__file__).resolve().parent
        status["torch_cuda"] = torch.version.cuda
        status["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - import failures are the point of this probe
        status["import_error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["vllm_path"] = str(package_dir)
    registry = package_dir / "model_executor" / "models" / "registry.py"
    if registry.exists():
        text = registry.read_text(encoding="utf-8", errors="replace")
        status["wemm_registered"] = "WeMMVideoMoeForConditionalGeneration" in text
        status["mtp_registered"] = "Qwen3_5MoeMTP" in text
    return status


def collect(model: str | None = None) -> dict:
    gpus = gpu_inventory()
    primary = gpus[0] if gpus else {}
    rc, smi_version, _ = run_cmd(["nvidia-smi", "--version"], timeout=20)
    torch_version = _version("torch")
    env = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch_version,
        "cuda": _version("nvidia-cuda-runtime-cu12") or _version("nvidia-cuda-runtime-cu13"),
        "vllm": _version("vllm"),
        "flashinfer": _version("flashinfer-python") or _version("flashinfer"),
        "transformers": _version("transformers"),
        "huggingface_hub": _version("huggingface-hub"),
        "gpu_name": primary.get("name"),
        "gpu_memory_total_mib": primary.get("memory_total_mib"),
        "gpu_memory_used_mib": primary.get("memory_used_mib"),
        "driver": primary.get("driver"),
        "compute_cap": primary.get("compute_cap"),
        "nvidia_smi_version": _first_line(smi_version) if rc == 0 else None,
        "gpu_count": len(gpus),
        "free_disk_gib": round(free_disk_gib(), 1),
        "ffmpeg": which("ffmpeg"),
        "model": model,
    }
    env.update(_vllm_patch_status())
    env["cuda"] = env.get("cuda") or env.get("torch_cuda")
    env["problems"] = _problems(env)
    return env


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _problems(env: dict) -> list[str]:
    problems = []
    if env.get("gpu_count", 0) == 0:
        problems.append("no NVIDIA GPU visible (nvidia-smi failed)")
    if not env.get("vllm"):
        problems.append("vllm is not installed in this Python environment")
    if env.get("vllm") and not env.get("wemm_registered"):
        problems.append("WeMM patch not detected in the vllm package (registry.py)")
    if not env.get("flashinfer"):
        problems.append("flashinfer-python is not installed (FlashInfer backend unavailable)")
    if not env.get("ffmpeg"):
        problems.append("ffmpeg not on PATH (needed for media; text-only runs are fine)")
    if env.get("gpu_memory_total_mib") and env["gpu_memory_total_mib"] < 70000:
        problems.append(
            f"GPU has {env['gpu_memory_total_mib']} MiB; BF16 WeMM-35B needs about 70 GiB of weights"
        )
    return problems


def main() -> int:
    env = collect()
    print(json.dumps(env, indent=2))
    if env["problems"]:
        print("\nproblems:")
        for item in env["problems"]:
            print(f"  - {item}")
        return 1
    print("\nOK: environment looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
