"""Benchmark orchestrator.

Typical use on the GPU box:

    ./run_bench.sh dry-run          # show exact server commands, no GPU needed
    ./run_bench.sh prep-mtp         # build the MTP sidecar (~9 GiB download)
    ./run_bench.sh run              # baseline + optimised arms, then the report
    ./run_bench.sh run --full       # adds K=5/K=7 sweep, n-gram, eager diagnostic

Without a GPU:

    ./run_bench.sh self-test        # mock server, validates the whole pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import env_check
from .common import (
    RESULTS,
    ensure_results_dirs,
    gpu_inventory,
    human_secs,
    log,
    print_rule,
    utc_now,
    write_json,
)
from .measure import MeasureContext, check_gpu_idle, run_arm
from .mock_server import MockHandle
from .report import build_report
from .variants import (
    QUANT_METHODS,
    RunConfig,
    build_arms,
    render_commands,
    select_arms,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_bench",
        description="Decode-TPS benchmark for WeMM-Video-35B-A3B on a single GPU.",
    )
    parser.add_argument("--model", default="tencent/WeMM-Video-35BA3B")
    parser.add_argument("--served-name", default="wemm-35b")
    parser.add_argument("--arms", default="default", help="'default', 'all', or comma-separated arm names")
    parser.add_argument("--full", action="store_true", help="include the extra sweep/diagnostic arms")
    parser.add_argument("--quick", action="store_true", help="2 repetitions instead of 5")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="preferred server port; the harness advances to the next free port "
        "unless --strict-port is given",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="fail instead of moving to another port when --port is busy",
    )
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor-parallel size / number of GPUs (the official WeMM recipe uses 2)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--mtp-head", default=None, help="path to the MTP sidecar directory")
    parser.add_argument(
        "--quant-model",
        default=None,
        help="path to a quantised checkpoint; enables the <quant>*/mtp arms that need it",
    )
    parser.add_argument(
        "--gptq-model",
        default=None,
        dest="quant_model",
        help=argparse.SUPPRESS,  # deprecated alias for --quant-model
    )
    parser.add_argument(
        "--quant-method",
        default="awq",
        choices=tuple(QUANT_METHODS),
        help="weight-quantisation method to benchmark (default: %(default)s)",
    )
    parser.add_argument("--prepare-mtp", action="store_true", help="build the MTP sidecar before running")
    parser.add_argument("--only-prepare", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--startup-timeout", type=float, default=2400.0)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="skip the free-memory check (BF16 will still OOM if the GPU is shared)",
    )
    parser.add_argument(
        "--wait-gpu",
        type=float,
        default=0,
        metavar="SECONDS",
        help="wait up to SECONDS for an idle GPU before giving up",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="run even if torchcodec cannot be imported (servers will likely fail)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print server commands and exit")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing results")
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="with --report-only, include arms from earlier invocations in the same directory",
    )
    parser.add_argument("--self-test", action="store_true", help="use the built-in mock server (no GPU)")
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument(
        "--custom-flags",
        default=None,
        help="extra server flags appended as one ad-hoc arm named 'custom'",
    )
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(
        args.results_dir or (RESULTS / "selftest" if args.self_test else RESULTS)
    ).resolve()
    results_dir, _ = ensure_results_dirs(results_dir)

    cfg = RunConfig(
        model=args.model,
        served_name=args.served_name,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem,
        tp_size=args.tp,
        port=args.port,
        max_tokens=args.max_tokens,
        reps=args.reps if args.reps is not None else (1 if args.self_test else (2 if args.quick else 5)),
        warmup=args.warmup if args.warmup is not None else (0 if args.self_test else 2),
    )

    if args.report_only:
        meta_path = results_dir / "run_meta.json"
        only = None
        if not args.all_results and meta_path.exists():
            try:
                only = set(json.loads(meta_path.read_text()).get("arms") or [])
            except (OSError, ValueError):
                only = None
        build_report(results_dir, only_arms=only)
        return 0

    if args.prepare_mtp:
        from .make_mtp_head import build as build_mtp

        log("building MTP sidecar")
        build(out=results_dir / "mtp_head", force=True)
        if args.only_prepare:
            return 0

    head = Path(args.mtp_head).resolve() if args.mtp_head else (results_dir / "mtp_head").resolve()
    if args.mtp_head and not head.exists() and not args.dry_run:
        raise SystemExit(f"--mtp-head {args.mtp_head} does not exist")
    # Always set the path so MTP arms are built and can be reported on; whether
    # they can run is decided below.
    cfg.mtp_head = head

    if (
        args.quant_model
        and not args.dry_run
        and not args.self_test
        and not Path(args.quant_model).exists()
    ):
        raise SystemExit(f"--quant-model {args.quant_model} does not exist")
    cfg.quant_model = args.quant_model
    cfg.quant_method = args.quant_method

    arms = build_arms(cfg, full=args.full)
    if args.custom_flags:
        import shlex

        from .variants import Arm

        arms.append(
            Arm(
                name="custom",
                label="Operator-supplied flags",
                kind="opt",
                extra_flags=shlex.split(args.custom_flags),
                claim_exact="unknown",
                note=f"--custom-flags {args.custom_flags}",
            )
        )
    if args.list_arms:
        for arm in arms:
            suffix = ""
            if arm.requires_mtp and not head.exists():
                suffix = f"  [needs sidecar: {head}]"
            if arm.requires_quant and not cfg.quant_model:
                suffix += "  [needs --quant-model]"
            print(f"{arm.name:16s} {arm.kind:10s} {arm.label}{suffix}")
        return 0

    arms = select_arms(arms, args.arms)
    if not arms:
        raise SystemExit("no arms selected")

    # Resolve missing-prerequisite behaviour: explicit request -> error, default -> drop.
    if not args.self_test and not args.dry_run:
        explicit = args.arms not in (None, "default", "all")

        def drop_or_fail(attr: str, message: str) -> None:
            nonlocal arms
            missing = [a for a in arms if getattr(a, attr)]
            if not missing:
                return
            names = ", ".join(a.name for a in missing)
            full = f"arm(s) {names} {message}"
            if explicit:
                raise SystemExit(full)
            log(full)
            arms = [a for a in arms if not getattr(a, attr)]

        if not head.exists():
            drop_or_fail(
                "requires_mtp",
                "need the MTP sidecar, which does not exist at:\n  "
                + str(head)
                + "\nBuild it with ./run_bench.sh prep-mtp (downloads ~10 GB), or pass an "
                "existing --mtp-head PATH. To measure without MTP, use --arms without "
                "the mtp* names.",
            )
        if not cfg.quant_model:
            drop_or_fail(
                "requires_quant",
                f"need a {args.quant_method} checkpoint; pass --quant-model PATH to a "
                "quantised copy of tencent/WeMM-Video-35BA3B. The <quant>* arms measure "
                "that checkpoint.",
            )

    mtp_config_broken = False
    if not args.self_test and any(a.requires_mtp for a in arms):
        from .fix_vllm_config import eagle_status, status as vllm_config_status

        qwen_state, eagle_state = vllm_config_status(), eagle_status()
        if qwen_state == "buggy" or eagle_state == "buggy":
            mtp_config_broken = True
            log(
                f"MTP preflight: vLLM backports missing (rope={qwen_state}, "
                f"eagle={eagle_state}); run ./run_bench.sh fix-mtp-config"
            )

    if args.dry_run:
        print(render_commands(cfg, arms))
        print(f"# model: {cfg.model}")
        print(f"# quant model: {cfg.quant_model or '(none)'} ({cfg.quant_method})")
        print(f"# mtp head: {cfg.mtp_head}")
        print(f"# arms: {', '.join(a.name for a in arms)}")
        print(
            f"# estimated wall time: "
            f"{human_secs(len(arms) * (420 + cfg.reps * 4 * 8))}"
        )
        return 0

    if args.self_test:
        env = {"simulated": True, "gpu_name": "mock", "vllm": "mock", "flashinfer": "mock"}
        write_json(results_dir / "env.json", env)
    else:
        print_rule("environment")
        env = env_check.collect(cfg.model, cfg.port)
        env["gpus"] = gpu_inventory()
        write_json(results_dir / "env.json", env)
        print(json.dumps({k: env.get(k) for k in ("gpu_name", "gpu_memory_total_mib", "driver",
                                                  "compute_cap", "torch", "vllm", "flashinfer")}, indent=2))
        if env.get("problems"):
            for problem in env["problems"]:
                log(f"  WARNING: {problem}")
        if not env.get("gpu_count"):
            raise SystemExit("no GPU visible; run './run_bench.sh env' to inspect the environment")
        if not env.get("vllm"):
            raise SystemExit(
                "vllm is not importable from this Python; run with "
                "BENCH_PYTHON=/path/to/patched-venv/bin/python ./run_bench.sh run"
            )
        if mtp_config_broken and not args.skip_preflight:
            print()
            print(
                "MTP arms cannot start: vLLM 0.18.0 needs two backports.\n"
                "  1. transformers_utils/configs/qwen3_5_moe.py passes a list to\n"
                "     ignore_keys_at_rope_validation; transformers 5.2 expects a set.\n"
                "  2. v1/spec_decode/eagle.py does not recognise WeMM architectures, so it\n"
                "     reads target_model.config.image_token_index, which WeMM does not have\n"
                "     (it uses image_token_id).\n"
                "Apply both with:\n"
                "  ./run_bench.sh fix-mtp-config\n"
                "then rerun (or pass --skip-preflight to measure the other arms anyway)."
            )
            print()
            raise SystemExit("preflight: MTP config backport missing")
        if env.get("torchcodec_ok") is False:
            print()
            print(env_check.torchcodec_fix(env))
            print()
            if not args.skip_preflight:
                raise SystemExit(
                    "preflight failed: torchcodec cannot load its shared libraries. "
                    "Fix it as above (or pass --skip-preflight to measure anyway)."
                )

    if not args.self_test:
        check_gpu_idle(args.allow_busy_gpu, args.wait_gpu, cfg.tp_size)

    print_rule("plan")
    for arm in arms:
        print(f"  {arm.name:16s} {arm.label}")
    print(f"  reps={cfg.reps} warmup={cfg.warmup} max_tokens={cfg.max_tokens} "
          f"port={cfg.port} results={results_dir}")
    if not args.yes and not args.self_test and sys.stdin.isatty():
        answer = input("Start the run? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return 1

    launcher = None
    if args.self_test:
        def launcher(arm, log_path):  # noqa: ANN001
            handle = MockHandle(arm.name, Path(log_path))
            return handle

    ctx = MeasureContext(
        cfg=cfg,
        results_dir=results_dir,
        allow_busy_gpu=args.allow_busy_gpu,
        startup_timeout=args.startup_timeout if not args.self_test else 30,
        auto_port=not args.strict_port,
    )

    write_json(
        results_dir / "run_meta.json",
        {
            "started_at": utc_now(),
            "argv": sys.argv,
            "self_test": bool(args.self_test),
            "arms": [a.name for a in arms],
            "config": {
                "model": cfg.model,
                "quant_model": cfg.quant_model,
                "quant_method": cfg.quant_method,
                "served_name": cfg.served_name,
                "max_model_len": cfg.max_model_len,
                "tp_size": cfg.tp_size,
                "gpu_mem_util": cfg.gpu_mem_util,
                "reps": cfg.reps,
                "warmup": cfg.warmup,
                "max_tokens": cfg.max_tokens,
                "mtp_head": str(cfg.mtp_head) if cfg.mtp_head else None,
            },
        },
    )

    t_start = time.time()
    failures = []
    # A crashed run must not leave an older report looking like today's result.
    for name in ("report.md", "summary.json"):
        stale = results_dir / name
        if stale.exists():
            stale.unlink()
    for arm in arms:
        stale = results_dir / f"{arm.name}.json"
        if stale.exists():
            stale.unlink()
    for index, arm in enumerate(arms, start=1):
        print_rule(f"arm {index}/{len(arms)}: {arm.name}")
        print(f"  {arm.label}")
        if arm.note:
            print(f"  note: {arm.note}")
        try:
            record = run_arm(arm, ctx, launcher=launcher)
        except KeyboardInterrupt:
            log("interrupted; stopping the suite")
            break
        except Exception as exc:  # noqa: BLE001 - one arm must not kill the run
            record = {
                "arm": arm.to_dict(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
                "requests": [],
                "quality": [],
                "summary": {},
                "started_at": utc_now(),
                "finished_at": utc_now(),
            }
            write_json(results_dir / f"{arm.name}.json", record)
            log(f"  ARM CRASHED: {record['error']}")
        if record.get("status") != "ok":
            failures.append(arm.name)
        tps = (record.get("summary") or {}).get("decode_tps_median")
        if tps:
            log(f"  {arm.name}: {tps:.1f} tok/s decode")
        log(f"  elapsed: {human_secs(time.time() - t_start)}")

    print_rule("report")
    summary = build_report(results_dir, print_table=False, only_arms={a.name for a in arms})
    _print_summary(summary, results_dir)
    if failures:
        log(f"arms that failed: {', '.join(failures)} (see logs in {results_dir / 'logs'})")
    return 0 if summary.get("baseline_tps") else 1


def _print_summary(summary: dict, results_dir: Path) -> None:
    print(f"baseline: {summary.get('baseline_tps') or float('nan'):.1f} tok/s")
    print(f"{'arm':18s} {'tok/s':>9s} {'speedup':>8s} {'exact':>7s}")
    for row in summary.get("arms", []):
        tps = row.get("decode_tps_median")
        speed = row.get("speedup_vs_baseline")
        exact = row.get("quality_exact_rate")
        print(
            f"{row['arm']:18s} "
            f"{(f'{tps:.1f}' if tps else '–'):>9s} "
            f"{(f'{speed:.2f}x' if speed else '–'):>8s} "
            f"{(f'{exact * 100:.0f}%' if exact is not None else '–'):>7s}"
        )
    best = summary.get("best_speedup")
    if best:
        verdict = "MET" if summary.get("target_met") else "NOT MET"
        print(f"target 2.5x: {verdict} (best {best:.2f}x, arm {summary.get('best_arm')})")
    print(f"full report: {results_dir / 'report.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
