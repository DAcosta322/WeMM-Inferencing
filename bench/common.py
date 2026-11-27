"""Shared helpers: paths, JSON I/O, process/port utilities, small formatting."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_results_dirs(results_dir: Path | None = None) -> tuple[Path, Path]:
    """Create and return (results_dir, logs_dir)."""
    res = Path(results_dir) if results_dir else RESULTS
    logs = res / "logs"
    res.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return res, logs


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path | str):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def read_text_tail(path: Path | str, n_lines: int = 40) -> str:
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    return "".join(lines[-n_lines:]).strip()


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def describe_port_holder(port: int, host: str = "127.0.0.1") -> str:
    """Best-effort "who is listening on this port" for error messages.

    ``ss`` usually needs the same user (or root) to show the process name; when
    it cannot, we still report that the port is taken and by which address.
    """
    rc, out, _ = run_cmd(["ss", "-ltnpH", f"sport = :{port}"], timeout=5)
    if rc == 0 and out.strip():
        procs = re.findall(r'users:\(\("([^"]+)",pid=(\d+)', out)
        if procs:
            return ", ".join(f"{name} (pid {pid})" for name, pid in procs)
        first = out.strip().splitlines()[0]
        return " ".join(first.split()[:4])
    rc, out, _ = run_cmd(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=5
    )
    if rc == 0 and out.strip():
        lines = out.strip().splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 2:
                return f"{parts[0]} (pid {parts[1]})"
    return "unknown process"


def pick_port(preferred: int, host: str = "127.0.0.1") -> int:
    if port_is_free(preferred, host):
        return preferred
    for candidate in range(preferred + 1, preferred + 50):
        if port_is_free(candidate, host):
            return candidate
    raise RuntimeError(f"no free TCP port near {preferred}")


def which(name: str) -> str | None:
    from shutil import which as _which

    return _which(name)


def run_cmd(
    cmd: list[str], timeout: float = 60.0, env: dict | None = None
) -> tuple[int, str, str]:
    """Run a command, returning (returncode, stdout, stderr). Never raises on rc."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd)}"


def cuda_visible_devices() -> str | None:
    """Return the raw CUDA_VISIBLE_DEVICES value, or None when all GPUs are visible."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value


def gpu_query_id() -> str | None:
    """The value to pass to ``nvidia-smi --id=``, or None for every GPU.

    Slurm and most schedulers allocate GPUs by setting CUDA_VISIBLE_DEVICES.
    Querying nvidia-smi without ``--id`` would then list *other users'* GPUs,
    which breaks both the idle check and the per-arm GPU sampling.
    """
    cvd = cuda_visible_devices()
    if cvd is None or cvd == "-1":
        return cvd
    return cvd


def first_gpu_id(default: str = "0") -> str:
    """Identifier of the first allocated GPU for nvidia-smi."""
    cvd = cuda_visible_devices()
    if not cvd or cvd == "-1":
        return default
    return cvd.split(",")[0].strip() or default


def find_npp_lib_dir() -> str | None:
    """Locate CUDA NPP shared libraries installed as a Python wheel.

    torchcodec (imported by the WeMM processor) links against libnppicc.so.12,
    which PyTorch does not bundle. The nvidia-npp-cu12/cu13 wheels ship it under
    ``nvidia/npp/lib``; that directory has to be on LD_LIBRARY_PATH.
    """
    from importlib import metadata

    for dist_name in ("nvidia-npp-cu12", "nvidia-npp-cu13", "nvidia-cuda-npp"):
        try:
            dist = metadata.distribution(dist_name)
        except metadata.PackageNotFoundError:
            continue
        base = Path(dist.locate_file("nvidia/npp/lib"))
        if base.is_dir():
            return str(base)
    return None


def runtime_env(base: dict | None = None) -> dict:
    """Environment for child processes: venv bin on PATH, NPP libs on LD_LIBRARY_PATH."""
    env = dict(os.environ if base is None else base)
    npp = find_npp_lib_dir()
    if npp:
        parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
        if npp not in parts:
            env["LD_LIBRARY_PATH"] = ":".join([npp, *parts])
    venv_bin = str(Path(sys.executable).parent)
    path = [p for p in env.get("PATH", "").split(":") if p]
    if venv_bin not in path:
        env["PATH"] = ":".join([venv_bin, *path])
    return env


def gpu_inventory() -> list[dict]:
    """Query nvidia-smi once (allocated GPUs only). Returns [] when none are visible."""
    if cuda_visible_devices() == "-1":
        return []
    cmd = ["nvidia-smi"]
    target = gpu_query_id()
    if target:
        cmd.append(f"--id={target}")
    cmd += [
        "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    rc, out, _ = run_cmd(cmd, timeout=20)
    if rc != 0:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_used_mib": int(float(parts[3])),
                "driver": parts[4],
                "compute_cap": parts[5],
            }
        )
    return gpus


def human_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def human_secs(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 90:
        return f"{int(minutes)}m{int(sec):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m"


def stable_hash(text: str, length: int = 12) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum counter/gauge samples by metric name (labels collapsed)."""
    totals: dict[str, float] = {}
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        name, _labels, value = match.groups()
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return totals


def free_disk_gib(path: Path | str = ROOT) -> float:
    """Free space on the filesystem that would hold ``path`` (may not exist yet)."""
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    try:
        usage = os.statvfs(str(target))
        return usage.f_bavail * usage.f_frsize / (1024**3)
    except OSError:
        return float("nan")


def print_rule(title: str = "", char: str = "─") -> None:
    width = 78
    if not title:
        print(char * width)
        return
    pad = max(0, width - len(title) - 2)
    print(f"{char * 2} {title} {char * pad}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)
