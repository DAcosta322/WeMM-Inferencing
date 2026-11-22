"""Aggregate arm results into a Markdown report and a compact summary.json."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .common import RESULTS, ensure_results_dirs, log, read_json, utc_now, write_json

TARGET_SPEEDUP = 2.5


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}{suffix}"


def _load_arm_files(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name in ("summary.json", "env.json"):
            continue
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "arm" in data and "requests" in data:
            records.append(data)
    records.sort(key=lambda r: (r["arm"]["kind"] != "baseline", r["arm"]["name"]))
    return records


def _tps(record: dict) -> float | None:
    return (record.get("summary") or {}).get("decode_tps_median")


def _common_prefix(a: list[int] | None, b: list[int] | None) -> int | None:
    if a is None or b is None:
        return None
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def _quality_table(records: list[dict]) -> list[dict]:
    baseline = next((r for r in records if r["arm"]["kind"] == "baseline"), None)
    if baseline is None:
        return []
    base_by_id = {q["prompt_id"]: q for q in baseline.get("quality", [])}
    rows = []
    for record in records:
        if record is baseline:
            continue
        compared = 0
        exact = 0
        ratios: list[float] = []
        divergences = []
        for q in record.get("quality", []):
            if not q.get("ok"):
                continue
            base = base_by_id.get(q["prompt_id"])
            if not base or not base.get("ok"):
                continue
            compared += 1
            same = (q.get("text") or "").strip() == (base.get("text") or "").strip()
            exact += int(same)
            prefix = _common_prefix(q.get("output_token_ids"), base.get("output_token_ids"))
            if prefix is not None:
                denom = max(
                    len(q.get("output_token_ids") or []),
                    len(base.get("output_token_ids") or []),
                    1,
                )
                ratios.append(prefix / denom)
            if not same and len(divergences) < 2:
                divergences.append(
                    {
                        "prompt_id": q["prompt_id"],
                        "arm_text": (q.get("text") or "")[:220],
                        "baseline_text": (base.get("text") or "")[:220],
                    }
                )
        rows.append(
            {
                "arm": record["arm"]["name"],
                "compared": compared,
                "exact": exact,
                "exact_rate": (exact / compared) if compared else None,
                "token_prefix_ratio_median": statistics.median(ratios) if ratios else None,
                "divergences": divergences,
                "claim": record["arm"].get("claim_exact"),
            }
        )
    return rows


def build_report(
    results_dir: Path = RESULTS,
    print_table: bool = True,
    only_arms: set[str] | None = None,
) -> dict:
    results_dir, _ = ensure_results_dirs(results_dir)
    records = _load_arm_files(results_dir)
    if only_arms is not None:
        records = [r for r in records if r["arm"]["name"] in only_arms]
    if not records:
        raise SystemExit(f"no arm results found in {results_dir}")

    baseline = next((r for r in records if r["arm"]["kind"] == "baseline"), None)
    baseline_tps = _tps(baseline) if baseline else None
    quality_rows = _quality_table(records)
    quality_by_arm = {row["arm"]: row for row in quality_rows}

    rows = []
    for record in records:
        tps = _tps(record)
        speedup = (tps / baseline_tps) if (tps and baseline_tps) else None
        summary = record.get("summary") or {}
        spec = record.get("spec_acceptance")
        rows.append(
            {
                "arm": record["arm"]["name"],
                "label": record["arm"]["label"],
                "kind": record["arm"]["kind"],
                "status": record.get("status"),
                "decode_tps_median": tps,
                "speedup_vs_baseline": speedup,
                "decode_tps_short": ((summary.get("decode_tps_by_bucket") or {}).get("short") or {}).get("median"),
                "decode_tps_long": ((summary.get("decode_tps_by_bucket") or {}).get("long") or {}).get("median"),
                "ttft_s_median": summary.get("ttft_s_median"),
                "output_tokens_median": summary.get("output_tokens_median"),
                "prompt_tokens_median": summary.get("prompt_tokens_median"),
                "acceptance_rate": (spec or {}).get("acceptance_rate"),
                "mean_acceptance_length": (spec or {}).get("mean_acceptance_length"),
                "quality_exact_rate": (quality_by_arm.get(record["arm"]["name"]) or {}).get("exact_rate"),
                "requests_ok": summary.get("requests_ok"),
                "requests_total": summary.get("requests_total"),
                "error": record.get("error"),
            }
        )

    best = max(
        (row for row in rows if row["status"] == "ok" and row["speedup_vs_baseline"]),
        key=lambda r: r["speedup_vs_baseline"],
        default=None,
    )

    suite_note = _suite_note(records)
    env = _load_env(results_dir)
    snapshot = records[0].get("snapshot") or {}
    simulated = any(
        (r.get("snapshot") or {}).get("simulated") for r in records
    )

    lines: list[str] = []
    lines.append("# WeMM-Video-35B-A3B decode TPS benchmark")
    lines.append("")
    if simulated:
        lines.append(
            "> **SIMULATED RUN** — these numbers come from the built-in mock server "
            "(`--self-test`), not from a GPU. They validate the harness end to end only."
        )
        lines.append("")
    lines.append(f"Generated: {utc_now()}")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    if baseline_tps is None:
        lines.append("Baseline arm did not produce a usable TPS measurement; see the table below.")
    elif best is not None:
        verdict = "MET" if best["speedup_vs_baseline"] >= TARGET_SPEEDUP else "NOT MET"
        lines.append(
            f"- Baseline decode TPS: **{baseline_tps:.1f} tok/s**"
        )
        lines.append(
            f"- Best arm: **{best['arm']}** at **{best['decode_tps_median']:.1f} tok/s**, "
            f"**{best['speedup_vs_baseline']:.2f}×** vs baseline"
        )
        lines.append(f"- Target of {TARGET_SPEEDUP:.1f}×: **{verdict}**")
    lines.append("")
    lines.append(suite_note)
    lines.append("")

    lines.append("## Arms")
    lines.append("")
    lines.append(
        "| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | "
        "Accept. | Exact-out | Notes |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        note = row.get("error") or ""
        if row["kind"] == "baseline":
            note = note or "reference"
        if row.get("acceptance_rate") is not None:
            accept = f"{row['acceptance_rate'] * 100:.0f}%"
        else:
            accept = "–"
        exact = (
            f"{row['quality_exact_rate'] * 100:.0f}%"
            if row.get("quality_exact_rate") is not None
            else "–"
        )
        tps = _fmt(row["decode_tps_median"])
        speed = _fmt(row["speedup_vs_baseline"], 2, "×")
        ttft = _fmt((row["ttft_s_median"] or 0) * 1000 if row["ttft_s_median"] else None, 0)
        short = _fmt(row["decode_tps_short"])
        long = _fmt(row["decode_tps_long"])
        note = (note or "").replace("|", "/")[:120]
        lines.append(
            f"| `{row['arm']}` | {row['status']} | {tps} | {speed} | {short} | {long} | "
            f"{ttft} | {accept} | {exact} | {note} |"
        )
    lines.append("")
    lines.append(
        "Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all "
        "measured requests; short/long are the per-context medians."
    )
    lines.append("")

    prompt_rows = _per_prompt_speedups(records, baseline)
    if prompt_rows:
        lines.append("## Speedup by prompt")
        lines.append("")
        prompt_ids = list(prompt_rows[0]["ratios"])
        lines.append("| Arm | " + " | ".join(prompt_ids) + " | median |")
        lines.append("| --- | " + " | ".join("---:" for _ in prompt_ids) + " | ---: |")
        for row in prompt_rows:
            cells = " | ".join(_fmt(row["ratios"][pid], 2, "×") for pid in prompt_ids)
            lines.append(f"| `{row['arm']}` | {cells} | {_fmt(row['median'], 2, '×')} |")
        lines.append("")
        lines.append(
            "Ratio of per-prompt median decode TPS to the baseline's median for the same "
            "prompt, then the median across prompts. This is the most robust speedup "
            "estimate because it cancels prompt-specific effects."
        )
        lines.append("")

    if any(row.get("acceptance_rate") for row in rows):
        lines.append("## Speculative decoding")
        lines.append("")
        lines.append("| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for record in records:
            spec = record.get("spec_acceptance")
            if not spec:
                continue
            lines.append(
                f"| `{record['arm']['name']}` | {spec['acceptance_rate'] * 100:.1f}% | "
                f"{spec['mean_acceptance_length']:.2f} | {spec['drafts']:.0f} | "
                f"{spec['accepted_tokens']:.0f} |"
            )
        lines.append("")
        lines.append(
            "Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. "
            "A K=3 run at 70% acceptance approaches ~2.0× over the same non-speculative server."
        )
        lines.append("")
    lines.append("## Output quality vs baseline")
    lines.append("")
    lines.append(
        "Greedy, non-streaming, `seed=0`, thinking disabled. Token-level prefix ratio is "
        "computed with the server tokenizer; 100% means the two outputs are token-identical."
    )
    lines.append("")
    lines.append("| Arm | Claim | Exact matches | Token prefix (median) | Verdict |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in quality_rows:
        if row["compared"] == 0:
            continue
        rate = row["exact_rate"] or 0.0
        ratio = row["token_prefix_ratio_median"]
        if row["claim"] == "exact":
            verdict = "OK (identical)" if rate == 1.0 else "**UNEXPECTED divergence**"
        elif row["claim"] == "baseline":
            verdict = "reference"
        elif row["claim"] == "unknown":
            verdict = "not claimed"
        else:
            verdict = "exact" if rate == 1.0 else "diverges (expected risk)"
        lines.append(
            f"| `{row['arm']}` | {row['claim']} | {row['exact']}/{row['compared']} | "
            f"{_fmt((ratio or 0) * 100, 1, '%')} | {verdict} |"
        )
    lines.append("")

    diverging = [row for row in quality_rows if row["compared"] and (row["exact_rate"] or 0) < 1.0]
    for row in diverging:
        for div in row["divergences"][:1]:
            lines.append(f"<details><summary>{row['arm']} / {div['prompt_id']} divergence</summary>")
            lines.append("")
            lines.append("```")
            lines.append("baseline: " + div["baseline_text"])
            lines.append("arm:      " + div["arm_text"])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    lines.append("## What actually ran")
    lines.append("")
    lines.append(
        "Flags are requests; the server log is the ground truth. These values are parsed "
        "from each arm's log after the run."
    )
    lines.append("")
    lines.append("| Arm | Attention backend | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: |")
    for record in records:
        snap = record.get("snapshot") or {}
        spec = snap.get("speculative")
        spec_txt = "–"
        if spec:
            spec_txt = f"{spec.get('method', '?')} K={spec.get('num_speculative_tokens', '?')}"
        lines.append(
            f"| `{record['arm']['name']}` | {snap.get('attention_backend') or '?'} | "
            f"{snap.get('cuda_graphs')} | {spec_txt} | {snap.get('tensor_parallel') or '?'} | "
            f"{snap.get('kv_cache_dtype') or '?'} | {snap.get('gpu_blocks') or '–'} |"
        )
    lines.append("")

    warnings = []
    for record in records:
        for warning in (record.get("snapshot") or {}).get("warnings") or []:
            warnings.append(f"- `{record['arm']['name']}`: {warning}")
    if warnings:
        lines.append("## Server warnings worth reading")
        lines.append("")
        lines.extend(warnings[:15])
        lines.append("")

    if env:
        lines.append("## Environment")
        lines.append("")
        for key in ("gpu_name", "gpu_memory_total_mib", "driver", "compute_cap", "torch", "cuda", "vllm", "flashinfer", "python"):
            if env.get(key) not in (None, "", []):
                value = env[key]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, "
        "`ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`."
    )
    lines.append(
        "- TTFT = client time to first generated token; E2E = client time to the final "
        "content/finish chunk; output tokens from `usage.completion_tokens`."
    )
    lines.append(
        f"- {records[0]['config']['warmup']} discarded warmups per arm, "
        f"{records[0]['config']['reps']} measured repetitions per prompt, "
        "prompts interleaved by repetition to spread thermal drift."
    )
    lines.append(
        "- One server per arm, started fresh and stopped afterwards; "
        "prefix caching disabled for every arm."
    )
    lines.append("")

    report = "\n".join(lines) + "\n"
    (results_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "generated_at": utc_now(),
        "simulated": simulated,
        "baseline_tps": baseline_tps,
        "target_speedup": TARGET_SPEEDUP,
        "best_arm": best["arm"] if best else None,
        "best_speedup": best["speedup_vs_baseline"] if best else None,
        "target_met": bool(best and best["speedup_vs_baseline"] >= TARGET_SPEEDUP),
        "arms": rows,
        "quality": quality_rows,
        "environment": env,
    }
    write_json(results_dir / "summary.json", summary)
    if print_table:
        print("\n" + report)
        log(f"report written to {results_dir / 'report.md'}")
    return summary


def _per_prompt_speedups(records: list[dict], baseline: dict | None) -> list[dict]:
    if baseline is None:
        return []
    base = _median_tps_by_prompt(baseline)
    rows = []
    for record in records:
        if record is baseline:
            continue
        arm = _median_tps_by_prompt(record)
        ratios = {
            pid: (arm[pid] / base[pid])
            for pid in arm
            if pid in base and base[pid] and arm[pid]
        }
        if not ratios:
            continue
        rows.append(
            {
                "arm": record["arm"]["name"],
                "ratios": ratios,
                "median": statistics.median(ratios.values()),
            }
        )
    return rows


def _median_tps_by_prompt(record: dict) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in record.get("requests", []):
        if item.get("ok") and item.get("decode_tps"):
            grouped.setdefault(item.get("prompt_id", "?"), []).append(item["decode_tps"])
    return {pid: statistics.median(vals) for pid, vals in grouped.items() if vals}


def _suite_note(records: list[dict]) -> str:
    fps = {r.get("prompt_suite_fingerprint") for r in records}
    if len(fps) > 1:
        return (
            "> **Warning:** arms used different prompt suites; speedups across arms are "
            "not directly comparable."
        )
    configs = {json.dumps(r.get("config"), sort_keys=True) for r in records}
    if len(configs) > 1:
        return (
            "> **Note:** arm configurations differ in token limits / repetition counts; "
            "check the protocol section before comparing."
        )
    return ""


def _load_env(results_dir: Path) -> dict:
    path = results_dir / "env.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    build_report(Path(RESULTS))
