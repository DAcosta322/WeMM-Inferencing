"""Launch, probe and stop a patched-vLLM OpenAI server for a single arm."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .common import (
    ROOT,
    describe_port_holder,
    first_gpu_id,
    log,
    parse_prometheus,
    pick_port,
    port_is_free,
    read_error_excerpt,
    read_text_tail,
    runtime_env,
    which,
)


class ServerStartError(RuntimeError):
    def __init__(self, message: str, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


@dataclass
class ServerSnapshot:
    """What the server log says actually happened (ground truth, not flags)."""

    attention_backend: str | None = None
    backends: list[str] = field(default_factory=list)
    cuda_graphs: bool | None = None
    speculative: dict | None = None
    tensor_parallel: int | None = None
    kv_cache_dtype: str | None = None
    max_model_len: int | None = None
    gpu_blocks: int | None = None
    version: str | None = None
    warnings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    simulated: bool = False

    def to_dict(self) -> dict:
        return {
            "attention_backend": self.attention_backend,
            "backends": self.backends,
            "cuda_graphs": self.cuda_graphs,
            "speculative": self.speculative,
            "tensor_parallel": self.tensor_parallel,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_model_len": self.max_model_len,
            "gpu_blocks": self.gpu_blocks,
            "version": self.version,
            "warnings": self.warnings[:20],
            "evidence": self.evidence[:20],
            "simulated": self.simulated,
        }


def launcher_prefix() -> list[str]:
    """Return the argv prefix that starts the OpenAI API server.

    ``vllm serve`` is preferred; the module entrypoint is a fallback for
    environments where the console script is not on PATH.
    """
    override = os.environ.get("VLLM_BIN")
    if override:
        return [override, "serve"]
    # A venv's console scripts live next to its interpreter, which is where the
    # patched install puts `vllm`; this works even without activating the venv.
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.exists():
        return [str(sibling), "serve"]
    binary = which("vllm")
    if binary:
        return [binary, "serve"]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


def build_server_cmd(
    model: str,
    served_name: str,
    host: str,
    port: int,
    flags: list[str],
) -> list[str]:
    prefix = launcher_prefix()
    if prefix[-1] == "serve":
        head = prefix + [model]
    else:
        head = prefix + ["--model", model]
    return head + ["--served-model-name", served_name, "--host", host, "--port", str(port), *flags]


# Kept for backwards compatibility with earlier call sites.
def vllm_command() -> list[str]:
    return launcher_prefix()


class ServerHandle:
    """A running vLLM server plus its log and metric scrapes."""

    def __init__(
        self,
        model: str,
        served_name: str,
        flags: list[str],
        log_path: Path,
        port: int | None = None,
        host: str = "127.0.0.1",
        startup_timeout: float = 2400.0,
        env: dict | None = None,
        auto_port: bool = True,
    ) -> None:
        self.model = model
        self.served_name = served_name
        self.flags = list(flags)
        self.log_path = Path(log_path)
        self.host = host
        requested = port or 8000
        self.auto_port = auto_port
        self.requested_port = requested
        if port_is_free(requested, host):
            self.port = requested
        elif auto_port:
            self.port = pick_port(requested, host)
            log(
                f"port {requested} is busy ({describe_port_holder(requested, host)}); "
                f"using {self.port} instead"
            )
        else:
            # --strict-port: fail loudly in start() instead of moving.
            self.port = requested
        self.startup_timeout = startup_timeout
        self.base_url = f"http://{host}:{self.port}"
        self._extra_env = dict(env or {})
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self.snapshot = ServerSnapshot()
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self._start_metrics: dict[str, float] = {}
        self._stop_metrics: dict[str, float] = {}
        self._stop_snapshot_metrics: dict[str, float] = {}

    # -- lifecycle --------------------------------------------------------

    def command(self) -> list[str]:
        return build_server_cmd(
            self.model, self.served_name, self.host, self.port, self.flags
        )

    def start(self) -> None:
        if not port_is_free(self.port, self.host):
            if self.auto_port:
                previous = self.port
                self.port = pick_port(previous + 1, self.host)
                self.base_url = f"http://{self.host}:{self.port}"
                log(
                    f"port {previous} was taken between launch and start; "
                    f"using {self.port} instead"
                )
            else:
                holder = describe_port_holder(self.port, self.host)
                raise ServerStartError(
                    f"port {self.port} is already in use by {holder}. "
                    f"Free it, or pass a different --port (or drop --strict-port "
                    f"to let the harness advance automatically).",
                    self.log_path,
                )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("w", encoding="utf-8")
        env = runtime_env()
        env.update(self._extra_env)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        env.setdefault("PYTHONUNBUFFERED", "1")
        cmd = self.command()
        log(f"starting server on port {self.port}: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.started_at = time.time()

    def wait_ready(self) -> None:
        if self.proc is None:
            raise ServerStartError("server was not started", self.log_path)
        deadline = time.time() + self.startup_timeout
        next_report = time.time() + 30
        last_error = ""
        while time.time() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                excerpt = read_error_excerpt(self.log_path)
                raise ServerStartError(
                    f"server exited during startup with code {rc}.\n--- error excerpt ---\n{excerpt}",
                    self.log_path,
                )
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3)
                if resp.status_code == 200:
                    self.ready_at = time.time()
                    self.snapshot = self._parse_log()
                    log(
                        f"server ready in {self.ready_at - (self.started_at or self.ready_at):.1f}s "
                        f"(backend={self.snapshot.attention_backend}, "
                        f"cudagraph={self.snapshot.cuda_graphs}, "
                        f"spec={self.snapshot.speculative})"
                    )
                    return
                last_error = f"health returned HTTP {resp.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            if time.time() >= next_report:
                elapsed = time.time() - (self.started_at or time.time())
                log(f"  still loading ({elapsed:.0f}s, last: {last_error[:80]})")
                next_report = time.time() + 30
            time.sleep(2)
        excerpt = read_error_excerpt(self.log_path)
        raise ServerStartError(
            f"server did not become healthy within {self.startup_timeout:.0f}s.\n"
            f"--- error excerpt ---\n{excerpt}",
            self.log_path,
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace: float = 240.0) -> None:
        if self.proc is None:
            return
        self._stop_metrics = self.metrics()
        self._stop_snapshot_metrics = self._stop_metrics
        try:
            self.snapshot = self._merge_snapshots(self.snapshot, self._parse_log())
        except Exception:  # never let teardown hide a measurement
            pass
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.time() + grace
            while time.time() < deadline and self.proc.poll() is None:
                time.sleep(1.0)
            if self.proc.poll() is None:
                log("server did not exit on SIGTERM; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        # Free GPU memory before the next arm starts.
        time.sleep(3.0)
        if self._log_fh is not None:
            self._log_fh.flush()
            self._log_fh.close()
            self._log_fh = None

    # -- metrics ----------------------------------------------------------

    def metrics(self) -> dict[str, float]:
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=10)
            if resp.status_code != 200:
                return {}
            return parse_prometheus(resp.text)
        except requests.RequestException:
            return {}

    def log_tail(self, lines: int = 60) -> str:
        return read_text_tail(self.log_path, lines)

    # -- log analysis -----------------------------------------------------

    def _parse_log(self) -> ServerSnapshot:
        snapshot = ServerSnapshot()
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return snapshot
        lines = text.splitlines()

        for line in lines:
            low = line.lower()
            backend_match = re.search(r"using\s+([a-z0-9_]+)\s+backend", low)
            if backend_match:
                name = backend_match.group(1).upper()
                if name not in snapshot.backends:
                    snapshot.backends.append(name)
                if snapshot.attention_backend is None:
                    snapshot.attention_backend = name
                snapshot.evidence.append(line.strip()[:300])
            if "capturing cuda graph" in low or "cuda graph" in low:
                if "not" in low and "captur" in low:
                    snapshot.cuda_graphs = False
                else:
                    snapshot.cuda_graphs = True
                snapshot.evidence.append(line.strip()[:300])
            if "speculative" in low and ("decoding" in low or "config" in low):
                snapshot.evidence.append(line.strip()[:300])
            if "tensor_parallel_size" in low or "tensor parallel size" in low:
                match = re.search(r"tensor_parallel_size[=: ]+(\d+)", low)
                if match:
                    snapshot.tensor_parallel = int(match.group(1))
            if "kv_cache_dtype" in low or "kv cache dtype" in low:
                match = re.search(
                    r"kv[_ ]cache[_ ]dtype['\"]?\s*[=:]\s*['\"]?([a-z0-9_]+)", low
                )
                if match:
                    snapshot.kv_cache_dtype = match.group(1)
            if "gpu blocks" in low or "gpu_blocks" in low:
                match = re.search(r"gpu[_ ]blocks[=: ]+(\d+)", low)
                if match:
                    snapshot.gpu_blocks = int(match.group(1))
            if "max_model_len" in low:
                match = re.search(r"max_model_len[=: ]+(\d+)", low)
                if match:
                    snapshot.max_model_len = int(match.group(1))
            if line.startswith("INFO") and "vllm" in low and "version" in low:
                match = re.search(r"v?(\d+\.\d+\.\d+\S*)", line)
                if match and snapshot.version is None:
                    snapshot.version = match.group(1)
            if "warn" in low and len(snapshot.warnings) < 25:
                if any(
                    key in low
                    for key in (
                        "falling back",
                        "unsupported",
                        "not support",
                        "deprecat",
                        "ignoring",
                        "cannot",
                    )
                ):
                    snapshot.warnings.append(line.strip()[:300])

        for line in lines:
            match = re.search(r"num_speculative_tokens['\"]?\s*[:=]\s*(\d+)", line)
            method = re.search(r"['\"]method['\"]\s*[:=]\s*['\"]([\w-]+)", line)
            if match or (method and "speculat" in line.lower()):
                snapshot.speculative = snapshot.speculative or {}
                if match:
                    snapshot.speculative["num_speculative_tokens"] = int(match.group(1))
                if method and "speculat" in line.lower():
                    snapshot.speculative["method"] = method.group(1)
        return snapshot

    @staticmethod
    def _merge_snapshots(old: ServerSnapshot, new: ServerSnapshot) -> ServerSnapshot:
        merged = old
        for attr in (
            "attention_backend",
            "backends",
            "cuda_graphs",
            "speculative",
            "tensor_parallel",
            "kv_cache_dtype",
            "max_model_len",
            "gpu_blocks",
            "version",
        ):
            if getattr(merged, attr) in (None, {}, False) and getattr(new, attr) not in (
                None,
                {},
                False,
            ):
                setattr(merged, attr, getattr(new, attr))
        merged.evidence = (new.evidence + merged.evidence)[:20]
        merged.warnings = (new.warnings + merged.warnings)[:25]
        return merged

    # -- metric helpers ---------------------------------------------------

    def spec_acceptance(self) -> dict | None:
        start = self._start_metrics
        end = self._stop_snapshot_metrics or self.metrics()
        if not start or not end:
            return None

        def delta(name: str) -> float:
            return end.get(name, 0.0) - start.get(name, 0.0)

        drafts = delta("vllm:spec_decode_num_drafts_total")
        drafted = delta("vllm:spec_decode_num_draft_tokens_total")
        accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
        if drafts <= 0 or drafted <= 0:
            return None
        return {
            "drafts": drafts,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted,
            "mean_acceptance_length": 1.0 + accepted / drafts,
        }

    def set_metrics_baseline(self, snapshot: dict) -> None:
        self._start_metrics = snapshot


class GpuSampler(threading.Thread):
    """Sample nvidia-smi in the background while an arm is measured."""

    QUERY = "utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"

    def __init__(self, interval: float = 1.0, gpu_id: str | None = None) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu_id = gpu_id if gpu_id else first_gpu_id()
        self.samples: list[dict] = []
        # NB: must not be called ``_stop`` — threading.Thread._stop() is called
        # by join() and shadowing it turns teardown into a TypeError.
        self._stop_event = threading.Event()

    def run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            f"--query-gpu={self.QUERY}",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if proc.returncode == 0 and proc.stdout.strip():
                    parts = [p.strip() for p in proc.stdout.strip().split(",")]
                    if len(parts) >= 5:
                        self.samples.append(
                            {
                                "utilization_gpu": _num(parts[0]),
                                "memory_used_mib": _num(parts[1]),
                                "power_w": _num(parts[2]),
                                "clocks_sm_mhz": _num(parts[3]),
                                "temperature_c": _num(parts[4]),
                            }
                        )
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=5)
        keys = ["utilization_gpu", "memory_used_mib", "power_w", "clocks_sm_mhz", "temperature_c"]
        summary: dict[str, dict] = {}
        for key in keys:
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if values:
                summary[key] = {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                }
        return {"samples": len(self.samples), "summary": summary}


def _num(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def spec_config_json(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))
