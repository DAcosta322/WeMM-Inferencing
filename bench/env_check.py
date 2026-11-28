"""Environment probe: is this box (and this venv) able to run the benchmark?"""

from __future__ import annotations

import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path

from .common import (
    describe_port_holder,
    find_npp_lib_dir,
    free_disk_gib,
    gpu_inventory,
    port_is_free,
    run_cmd,
    runtime_env,
    which,
)

NPP_FIX = (
    "torchcodec cannot load its shared libraries (usually libnppicc.so.12, which\n"
    "PyTorch does not bundle; WeMM's processor imports torchcodec at model load).\n"
    "Fix, inside the serving venv:\n"
    "  python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87\n"
    "  # if pip is unavailable: uv pip install --python \"$(command -v python)\" \\\n"
    "  #   --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87\n"
    "  python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print(\"torchcodec ok\")'\n"
    "The harness injects the wheel's nvidia/npp/lib into LD_LIBRARY_PATH automatically,\n"
    "so no manual export is needed once the wheel is installed. If CUDA comes from a\n"
    "module instead, add $CUDA_HOME/targets/x86_64-linux/lib to LD_LIBRARY_PATH."
)


def torchcodec_fix(env: dict) -> str:
    error = (env.get("torchcodec_error") or "").lower()
    if "module named 'torchcodec'" in error or "no module named 'torchcodec'" in error:
        return (
            "torchcodec is not installed in this Python environment. The WeMM serving\n"
            "profile pins torchcodec==0.10.0+cu129; finish the installer\n"
            "  bash patch/install.sh vllm 0.18.0\n"
            "or install it directly:\n"
            "  python -m pip install torchcodec==0.10.0 \\\n"
            "    --index-url https://download.pytorch.org/whl/cu129"
        )
    return NPP_FIX


def _torchcodec_probe() -> tuple[bool, str | None]:
    """Import torchcodec the way the WeMM processor does, in a child process."""
    code = "from torchcodec.decoders import VideoDecoder, AudioDecoder"
    rc, out, err = run_cmd([sys.executable, "-c", code], timeout=240, env=runtime_env())
    if rc == 0:
        return True, None
    message = (err or out).strip()
    return False, message[-1500:] if message else f"exit code {rc}"


def _npp_fields() -> dict:
    from importlib import metadata as _metadata

    package = None
    for dist_name in ("nvidia-npp-cu12", "nvidia-npp-cu13", "nvidia-cuda-npp"):
        try:
            _metadata.distribution(dist_name)
            package = dist_name
            break
        except _metadata.PackageNotFoundError:
            continue
    lib_dir = find_npp_lib_dir()
    present = None
    if lib_dir:
        present = any(
            (Path(lib_dir) / name).exists()
            for name in ("libnppicc.so.12", "libnppicc.so.13")
        )
    return {"npp_package": package, "npp_lib_dir": lib_dir, "npp_lib_present": present}


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


def _backport_status() -> dict:
    """Report the three vLLM 0.18.0 backport statuses.

    Patched vLLM 0.31.0 carries the Qwen3.5-MoE rope fix natively, so on a
    0.31.0 install these read ``fixed``/``unknown`` (never ``buggy``) and no
    ``fix-mtp-config`` pass is required.
    """
    try:
        from .fix_vllm_config import eagle_status, modelopt_status, status as rope_status
    except Exception:  # noqa: BLE001 - a broken vllm install must not hide other probes
        return {}
    rope = rope_status()
    eagle = eagle_status()
    modelopt = modelopt_status()
    return {
        "rope_backport": rope,
        "eagle_backport": eagle,
        "modelopt_backport": modelopt,
        "mtp_backports_needed": rope == "buggy" or eagle == "buggy",
        "nvfp4_backport_needed": modelopt == "buggy",
    }


def collect(model: str | None = None, port: int = 8000) -> dict:
    gpus = gpu_inventory()
    _port_free = port_is_free(port)
    primary = gpus[0] if gpus else {}
    rc, smi_version, _ = run_cmd(["nvidia-smi", "--version"], timeout=20)
    torch_version = _version("torch")
    from . import BUILD, __version__

    env = {
        "harness_version": f"{__version__} ({BUILD})",
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
        "gpu_memory_free_mib": (
            primary.get("memory_total_mib", 0) - primary.get("memory_used_mib", 0)
            if primary
            else None
        ),
        "driver": primary.get("driver"),
        "compute_cap": primary.get("compute_cap"),
        "nvidia_smi_version": _first_line(smi_version) if rc == 0 else None,
        "gpu_count": len(gpus),
        "free_disk_gib": _round_free(free_disk_gib()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": _hf_home(),
        "hf_home_free_gib": _round_free(free_disk_gib(_hf_home())),
        "ffmpeg": which("ffmpeg"),
        "model": model,
        "bench_port": port,
        "bench_port_free": _port_free,
        "bench_port_holder": None if _port_free else describe_port_holder(port),
    }
    env.update(_vllm_patch_status())
    env.update(_backport_status())
    env["cuda"] = env.get("cuda") or env.get("torch_cuda")
    env.update(_npp_fields())
    codec_ok, codec_error = _torchcodec_probe()
    env["torchcodec_ok"] = codec_ok
    env["torchcodec_error"] = codec_error
    env["problems"] = _problems(env)
    return env


def _round_free(value: float) -> float | None:
    return round(value, 1) if value == value else None


def _hf_home() -> str:
    """HF cache root (honours HF_HOME, then the huggingface_hub default)."""
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return explicit
    hf_hub = os.environ.get("HF_HUB_CACHE")
    if hf_hub:
        return str(Path(hf_hub).parent)
    return str(Path.home() / ".cache" / "huggingface")


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
    if env.get("vllm") and env.get("mtp_backports_needed"):
        problems.append(
            "vLLM MTP backports are missing (rope/eagle); run "
            "./run_bench.sh fix-mtp-config, or use patched vLLM 0.31.0 which "
            "carries the Qwen3.5-MoE fixes natively"
        )
    if not env.get("flashinfer"):
        problems.append("flashinfer-python is not installed (FlashInfer backend unavailable)")
    if env.get("bench_port_free") is False:
        problems.append(
            f"TCP port {env.get('bench_port')} is already in use by "
            f"{env.get('bench_port_holder')}; the harness will auto-advance to the "
            "next free port (or pass --port)"
        )
    if not env.get("ffmpeg"):
        problems.append("ffmpeg not on PATH (TorchCodec in the WeMM patch expects it)")
    hf_free = env.get("hf_home_free_gib")
    if isinstance(hf_free, (int, float)) and hf_free == hf_free and hf_free < 100:
        problems.append(
            f"HF cache {env.get('hf_home')} has only {env['hf_home_free_gib']} GiB free; "
            "the BF16 checkpoint plus the MTP sidecar needs ~80 GiB"
        )
    if env.get("torchcodec_ok") is False:
        error = (env.get("torchcodec_error") or "").lower()
        if "module named 'torchcodec'" in error:
            problems.append(
                "torchcodec is not installed — WeMM's processor imports it at model "
                "load, so every server will fail; see ./run_bench.sh env"
            )
        elif env.get("npp_lib_present") is False:
            problems.append(
                f"{env.get('npp_package')} is installed but libnppicc.so.12 was not "
                f"found under {env.get('npp_lib_dir')}; reinstall the wheel"
            )
        else:
            problems.append(
                "torchcodec is not loadable (libnppicc.so.12 missing) — every server "
                "will fail at startup; see ./run_bench.sh env"
            )
    if env.get("gpu_memory_total_mib") and env["gpu_memory_total_mib"] < 70000:
        problems.append(
            f"GPU has {env['gpu_memory_total_mib']} MiB; BF16 WeMM-35B needs about 70 GiB of weights"
        )
    free = env.get("gpu_memory_free_mib")
    if free is not None and free < 78000:
        problems.append(
            f"only {free} MiB of GPU memory is free; BF16 + MTP needs about 78,000 MiB. "
            "The run will refuse to start until the GPU is free (see the GPU-busy message "
            "for options; do not kill another user's process)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe the benchmark environment.")
    parser.add_argument("--port", type=int, default=8000, help="preferred benchmark port")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    env = collect(model=args.model, port=args.port)
    print(json.dumps(env, indent=2))
    if env["problems"]:
        print("\nproblems:")
        for item in env["problems"]:
            print(f"  - {item}")
        if env.get("torchcodec_ok") is False:
            print()
            print(torchcodec_fix(env))
            if env.get("torchcodec_error"):
                print("\nlast torchcodec error (tail):")
                for line in env["torchcodec_error"].splitlines()[-12:]:
                    print(f"  {line}")
        return 1
    print("\nOK: environment looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
