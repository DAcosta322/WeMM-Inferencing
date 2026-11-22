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


def run_cmd(cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """Run a command, returning (returncode, stdout, stderr). Never raises on rc."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd)}"


def gpu_inventory() -> list[dict]:
    """Query nvidia-smi once. Returns [] when no driver/GPU is visible."""
    rc, out, _ = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
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
    try:
        usage = os.statvfs(str(path))
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
