"""Measure one arm: launch server, warm up, check quality, measure, tear down."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .client import ChatClient
from .common import (
    ensure_results_dirs,
    gpu_inventory,
    log,
    read_text_tail,
    run_cmd,
    utc_now,
    write_json,
)
from .prompts import bench_suite, quality_suite, suite_fingerprint
from .server import GpuSampler, ServerHandle, ServerStartError
from .variants import Arm, RunConfig


@dataclass
class MeasureContext:
    cfg: RunConfig
    results_dir: Path
    allow_busy_gpu: bool = False
    startup_timeout: float = 2400.0
    read_timeout: float = 1200.0
    warmup_tokens: int = 32
    auto_port: bool = True
    inter_request_sleep: float = 0.2
    env: dict = field(default_factory=dict)


def _gpu_compute_processes() -> list[str]:
    rc, out, _ = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if rc != 0 or not out.strip():
        return []
    lines = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            lines.append(f"pid {parts[0]} {parts[1]} ({parts[2]} MiB)")
    return lines


REQUIRED_FREE_MIB = 78_000  # BF16 weights (~69 GiB) + MTP sidecar + KV/graphs


def check_gpu_idle(allow_busy: bool, wait_seconds: float = 0) -> list[dict]:
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        gpus = gpu_inventory()
        if not gpus:
            return gpus
        busy = [g for g in gpus if g["memory_used_mib"] > 3000]
        if not busy or allow_busy:
            return gpus
        if time.time() >= deadline:
            break
        free = busy[0]["memory_total_mib"] - busy[0]["memory_used_mib"]
        log(
            f"waiting for an idle GPU: {free} MiB free, need ~{REQUIRED_FREE_MIB} MiB; "
            f"{int(deadline - time.time())}s left"
        )
        time.sleep(30)
    processes = _gpu_compute_processes()
    lines = []
    for gpu in busy:
        free = gpu["memory_total_mib"] - gpu["memory_used_mib"]
        lines.append(
            f"gpu{gpu['index']} ({gpu['name']}): {free} MiB free of "
            f"{gpu['memory_total_mib']} MiB ({gpu['memory_used_mib']} MiB in use)"
        )
    detail = "\n  ".join(lines)
    holders = ""
    if processes:
        holders = "\nGPU processes:\n  " + "\n  ".join(processes[:8])
    raise SystemExit(
        f"Not enough free GPU memory to start the benchmark:\n  {detail}{holders}\n"
        f"BF16 weights + the MTP sidecar + KV/activations need about "
        f"{REQUIRED_FREE_MIB} MiB free.\n"
        "\n"
        "Do NOT kill a PID you do not own. Options, in order:\n"
        "  1. If the process is yours, stop it (kill <pid>) and rerun.\n"
        "  2. Wait for the other job to finish, or request another allocation/GPU.\n"
        "     Inspect this allocation with:\n"
        "       srun --gres=gpu:1 --mem=8G --pty bash -lc 'nvidia-smi; echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'\n"
        "     If the node enables MPS, Slurm may share a GPU across users; ask for a\n"
        "     free node or exclusive GPU access.\n"
        "  3. --allow-busy-gpu overrides this check, but BF16 will not fit next to a\n"
        "     37 GiB neighbour and vLLM will OOM during startup. It is only useful for\n"
        "     small models or quantised checkpoints.\n"
    )


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _quantile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    low = int(pos)
    high = min(low + 1, len(clean) - 1)
    frac = pos - low
    return clean[low] * (1 - frac) + clean[high] * frac


def _environment_hint(text: str) -> str | None:
    low = text.lower()
    if "libnppicc" in low or "torchcodec" in low or "could not load this library" in low:
        return (
            "torchcodec/NPP: install nvidia-npp-cu12==12.4.1.87 in the serving venv; "
            "the harness then injects LD_LIBRARY_PATH automatically. "
            "Run ./run_bench.sh env for the exact commands."
        )
    if "flashinfer" in low and ("not valid" in low or "no kernel" in low):
        return (
            "FlashInfer rejected on this GPU; rerun the arm without "
            "--attention-backend FLASHINFER (the harness retries once automatically)."
        )
    if "out of memory" in low or "cuda oom" in low:
        return "GPU OOM: retry with --gpu-mem 0.95 or a smaller --max-model-len."
    return None


def _without_flag(flags: list[str], flag: str) -> list[str]:
    """Return flags with ``flag`` and its value removed."""
    out: list[str] = []
    skip = False
    for item in flags:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        out.append(item)
    return out


def summarise_requests(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("ok") and r.get("decode_tps")]
    tps = [r["decode_tps"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    e2e = [r["e2e_s"] for r in ok if r.get("e2e_s") is not None]
    out_tokens = [r["output_tokens"] for r in ok if r.get("output_tokens")]
    prompt_tokens = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    summary = {
        "requests_total": len(results),
        "requests_ok": len(ok),
        "decode_tps_median": _median(tps),
        "decode_tps_mean": (sum(tps) / len(tps)) if tps else None,
        "decode_tps_p10": _quantile(tps, 0.10),
        "decode_tps_p90": _quantile(tps, 0.90),
        "ttft_s_median": _median(ttft),
        "e2e_s_median": _median(e2e),
        "output_tokens_median": _median(out_tokens),
        "prompt_tokens_median": _median(prompt_tokens),
        "decode_tps_by_bucket": {},
    }
    for bucket in ("short", "long"):
        values = [r["decode_tps"] for r in ok if r.get("bucket") == bucket]
        summary["decode_tps_by_bucket"][bucket] = {
            "median": _median(values),
            "n": len(values),
        }
    return summary


def run_arm(
    arm: Arm,
    ctx: MeasureContext,
    launcher: Callable[[Arm, Path], ServerHandle] | None = None,
) -> dict:
    results_dir, logs_dir = ensure_results_dirs(ctx.results_dir)
    log_path = logs_dir / f"{arm.name}.log"
    cfg = ctx.cfg
    record: dict = {
        "arm": arm.to_dict(),
        "status": "pending",
        "started_at": utc_now(),
        "prompt_suite_fingerprint": suite_fingerprint(bench_suite()),
        "config": {
            "model": cfg.model,
            "served_name": cfg.served_name,
            "max_model_len": cfg.max_model_len,
            "tp_size": cfg.tp_size,
            "gpu_mem_util": cfg.gpu_mem_util,
            "max_tokens": cfg.max_tokens,
            "reps": cfg.reps,
            "warmup": cfg.warmup,
            "mtp_head": str(cfg.mtp_head) if cfg.mtp_head else None,
        },
        "gpus": gpu_inventory(),
        "requests": [],
        "quality": [],
        "metrics": {},
        "snapshot": {},
        "error": None,
        "log_path": str(log_path),
    }

    handle: ServerHandle | None = None
    client: ChatClient | None = None
    gpu_sampler: GpuSampler | None = None
    t_arm = time.time()
    try:
        def launch(flags: list[str]):
            if launcher is None:
                new_handle = ServerHandle(
                    model=cfg.model,
                    served_name=cfg.served_name,
                    flags=flags,
                    log_path=log_path,
                    host=cfg.host,
                    port=cfg.port,
                    startup_timeout=ctx.startup_timeout,
                    env=ctx.env,
                    auto_port=ctx.auto_port,
                )
            else:
                new_handle = launcher(arm, log_path)
            try:
                new_handle.start()
                new_handle.wait_ready()
            except Exception:
                # A timed-out server may still be holding GPU memory; stop it
                # before the caller retries with different flags.
                try:
                    new_handle.stop()
                except Exception:  # noqa: BLE001 - teardown must not mask the error
                    pass
                raise
            return new_handle

        try:
            handle = launch(arm.flags(cfg))
        except ServerStartError as exc:
            low = str(exc).lower()
            # An explicit attention backend can crash engine init even when the
            # auto-selected backend is the same one (seen with FLASHINFER on a
            # rebuilt environment). Retry once with vLLM's own selection.
            if "--attention-backend" in arm.flags(cfg) and any(
                marker in low
                for marker in (
                    "engine core initialization failed",
                    "flashinfer",
                    "not valid",
                    "not support",
                    "importerror",
                )
            ):
                fallback = _without_flag(arm.flags(cfg), "--attention-backend")
                record["fallback"] = (
                    "explicit --attention-backend failed at engine init; "
                    "retried with vLLM's automatic backend selection"
                )
                log(f"  {record['fallback']}; retrying once")
                if handle is not None:
                    handle.stop()
                handle = launch(fallback)
            else:
                raise
        record["startup_s"] = round(time.time() - t_arm, 2)
        record["snapshot"] = handle.snapshot.to_dict() if hasattr(handle.snapshot, "to_dict") else dict(handle.snapshot)

        client = ChatClient(handle.base_url, cfg.served_name, read_timeout=ctx.read_timeout)
        if not client.health():
            raise ServerStartError("server reports unhealthy right after startup", log_path)

        prompts = bench_suite()
        q_prompts = quality_suite()

        # --- warmup (discarded) ----------------------------------------
        warmup_failures = 0
        for _ in range(max(0, cfg.warmup)):
            warm = client.stream_chat(
                prompts[0].text,
                max_tokens=ctx.warmup_tokens,
                ignore_eos=False,
                prompt_id="warmup",
                arm=arm.name,
            )
            if not warm.ok:
                warmup_failures += 1
                log(f"  warmup failed: {warm.error}")
        record["warmup_failures"] = warmup_failures
        if warmup_failures >= max(2, cfg.warmup):
            raise ServerStartError("every warmup request failed", log_path)

        # --- quality (non-streaming, greedy) ---------------------------
        for prompt in q_prompts:
            q = client.chat(prompt.text, max_tokens=192, prompt_id=prompt.id, arm=arm.name)
            entry = q.to_dict()
            entry["bucket"] = prompt.bucket
            entry["prompt_text"] = prompt.text
            entry["output_token_ids"] = client.token_ids(q.text) if q.ok else None
            record["quality"].append(entry)
            log(
                f"  quality {prompt.id}: {'ok' if q.ok else 'FAIL'} "
                f"({len(q.text)} chars)"
            )

        # --- measured loop ---------------------------------------------
        handle.set_metrics_baseline(handle.metrics())
        gpu_sampler = GpuSampler()
        gpu_sampler.start()

        consecutive_failures = 0
        for rep in range(cfg.reps):
            for prompt in prompts:
                if not handle.is_alive():
                    raise ServerStartError("server died during measurement", log_path)
                result = client.stream_chat(
                    prompt.text,
                    max_tokens=cfg.max_tokens,
                    ignore_eos=True,
                    prompt_id=f"{prompt.id}",
                    arm=arm.name,
                )
                item = result.to_dict()
                item["bucket"] = prompt.bucket
                item["rep"] = rep
                record["requests"].append(item)
                if not result.ok:
                    consecutive_failures += 1
                    log(f"  {prompt.id} rep{rep}: FAIL {result.error}")
                    if consecutive_failures >= 3:
                        raise ServerStartError(
                            "three consecutive requests failed", log_path
                        )
                else:
                    consecutive_failures = 0
                    tps_text = f"{result.decode_tps:.1f}" if result.decode_tps else "n/a"
                    log(
                        f"  {prompt.id} rep{rep}: {tps_text} tok/s "
                        f"(ttft {(result.ttft_s or 0) * 1000:.0f} ms, "
                        f"{result.output_tokens} tok, {result.prompt_tokens} prompt tok)"
                    )
                time.sleep(ctx.inter_request_sleep)

        record["metrics"] = handle.metrics()
        record["spec_acceptance"] = handle.spec_acceptance() if hasattr(handle, "spec_acceptance") else None
        record["summary"] = summarise_requests(record["requests"])
        ok_count = record["summary"]["requests_ok"]
        record["status"] = "ok" if ok_count > 0 else "no_success"
    except ServerStartError as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        record["log_tail"] = read_text_tail(log_path, 80)
        hint = _environment_hint(str(exc) + "\n" + record["log_tail"])
        if hint:
            record["hint"] = hint
            log(f"  HINT: {hint}")
        log(f"  ARM FAILED: {exc}")
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    except Exception as exc:  # noqa: BLE001 - keep the suite running
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["log_tail"] = read_text_tail(log_path, 80)
        log(f"  ARM ERROR: {record['error']}")
    finally:
        # Teardown is best-effort: a broken sampler or a stuck server must not
        # discard a measurement that already completed.
        if gpu_sampler is not None:
            try:
                record["gpu_sampling"] = gpu_sampler.stop()
            except Exception as exc:  # noqa: BLE001
                record["gpu_sampling_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  WARNING: GPU sampler teardown failed: {exc}")
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        if handle is not None:
            try:
                if hasattr(handle, "set_metrics_baseline"):
                    record["spec_acceptance"] = (
                        handle.spec_acceptance() if hasattr(handle, "spec_acceptance") else None
                    )
                handle.stop()
                if hasattr(handle, "snapshot"):
                    snapshot = (
                        handle.snapshot.to_dict()
                        if hasattr(handle.snapshot, "to_dict")
                        else dict(handle.snapshot)
                    )
                    record["snapshot"] = snapshot
            except Exception as exc:  # noqa: BLE001
                record.setdefault("teardown_error", f"{type(exc).__name__}: {exc}")
                log(f"  WARNING: server teardown failed: {exc}")
        if not record.get("summary"):
            record["summary"] = summarise_requests(record.get("requests", []))
        record["duration_s"] = round(time.time() - t_arm, 2)
        record["finished_at"] = utc_now()
        out_path = results_dir / f"{arm.name}.json"
        write_json(out_path, record)
        log(f"  wrote {out_path}")
    return record
