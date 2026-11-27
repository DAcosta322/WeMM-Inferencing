"""Arm definitions: exactly which server flags each measurement runs with.

Arms are deliberately small and additive so that the contribution of each
optimisation is visible:

    baseline          official WeMM recipe, BF16, TP=1, nothing else
    flashinfer        + --attention-backend FLASHINFER  (CUDA graphs are on by default)
    flashinfer_fp8kv  + --kv-cache-dtype fp8
    mtp3              flashinfer + Qwen3.5 MTP head, K=3
    mtp3_fp8kv        flashinfer + fp8 KV + MTP K=3   (all listed optimisations)
    mtp5*             flashinfer + MTP K=5            (* --full)
    ngram*            flashinfer + n-gram speculation (* --full, no extra weights)
    eager*            baseline with CUDA graphs off   (* --full, diagnostic only)

Every arm runs the same prompts, seeds, token limits and measurement protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    model: str = "tencent/WeMM-Video-35BA3B"
    served_name: str = "wemm-35b"
    max_model_len: int = 8192
    tp_size: int = 1
    gpu_mem_util: float = 0.90
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens: int = 256
    reps: int = 5
    warmup: int = 2
    mtp_head: Path | None = None

    def common_flags(self) -> list[str]:
        return [
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_model_len),
            "--no-enable-prefix-caching",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--tensor-parallel-size",
            str(self.tp_size),
            "--gpu-memory-utilization",
            f"{self.gpu_mem_util:.2f}",
            "--limit-mm-per-prompt",
            '{"image":4,"video":2,"audio":2}',
            "--media-io-kwargs",
            '{"video":{"fps":2,"min_frames":4,"max_frames":16,"total_pixels":262144}}',
        ]


@dataclass
class Arm:
    name: str
    label: str
    kind: str  # baseline | opt | diagnostic
    extra_flags: list[str] = field(default_factory=list)
    claim_exact: str = "exact"  # exact | likely | no
    note: str = ""
    requires_mtp: bool = False

    def flags(self, cfg: RunConfig) -> list[str]:
        return cfg.common_flags() + self.extra_flags

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "extra_flags": self.extra_flags,
            "claim_exact": self.claim_exact,
            "note": self.note,
            "requires_mtp": self.requires_mtp,
        }


def _mtp_spec(head: Path | None, k: int) -> list[str]:
    if head is None:
        raise ValueError("MTP arm requested but no --mtp-head path is set")
    payload = {
        "model": str(head),
        "method": "mtp",
        "num_speculative_tokens": k,
    }
    return ["--speculative-config", json.dumps(payload, separators=(",", ":"))]


def build_arms(cfg: RunConfig, full: bool = False) -> list[Arm]:
    flashinfer = ["--attention-backend", "FLASHINFER"]
    fp8kv = ["--kv-cache-dtype", "fp8"]

    arms = [
        Arm(
            name="baseline",
            label="Out-of-the-box BF16 (official recipe, TP=1)",
            kind="baseline",
            claim_exact="baseline",
            note="No extra flags. CUDA graphs and the default attention backend are whatever vLLM picks.",
        ),
        Arm(
            name="flashinfer",
            label="+ FlashInfer pinned",
            kind="opt",
            extra_flags=flashinfer,
            claim_exact="exact",
            note="CUDA graphs are already enabled by default; this pins the attention backend.",
        ),
        Arm(
            name="flashinfer_fp8kv",
            label="+ FlashInfer + FP8 KV cache",
            kind="opt",
            extra_flags=flashinfer + fp8kv,
            claim_exact="likely",
            note="FP8 KV is lossy by construction; the quality table shows whether tokens moved.",
        ),
    ]

    mtp_ready = cfg.mtp_head is not None
    if mtp_ready:
        arms.append(
            Arm(
                name="mtp3",
                label="+ FlashInfer + MTP K=3",
                kind="opt",
                extra_flags=flashinfer + _mtp_spec(cfg.mtp_head, 3),
                claim_exact="exact",
                note="Grafted Qwen3.5 MTP head; speculative verification keeps greedy output identical.",
                requires_mtp=True,
            )
        )
        arms.append(
            Arm(
                name="mtp3_fp8kv",
                label="+ FlashInfer + FP8 KV + MTP K=3",
                kind="opt",
                extra_flags=flashinfer + fp8kv + _mtp_spec(cfg.mtp_head, 3),
                claim_exact="likely",
                note="The full listed optimisation set.",
                requires_mtp=True,
            )
        )
        if full:
            arms.append(
                Arm(
                    name="flashinfer_fp8kv_scaled",
                    label="+ FlashInfer + FP8 KV (runtime scales)",
                    kind="opt",
                    extra_flags=flashinfer + fp8kv + ["--calculate-kv-scales"],
                    claim_exact="likely",
                    note="Adds --calculate-kv-scales; better FP8 accuracy, one calibration pass.",
                )
            )
            arms.append(
                Arm(
                    name="mtp5",
                    label="+ FlashInfer + MTP K=5",
                    kind="opt",
                    extra_flags=flashinfer + _mtp_spec(cfg.mtp_head, 5),
                    claim_exact="exact",
                    note="Deeper speculation trades acceptance for fewer target forwards.",
                    requires_mtp=True,
                )
            )
            arms.append(
                Arm(
                    name="mtp7",
                    label="+ FlashInfer + MTP K=7",
                    kind="opt",
                    extra_flags=flashinfer + _mtp_spec(cfg.mtp_head, 7),
                    claim_exact="exact",
                    note="Upper end of the acceptance sweep.",
                    requires_mtp=True,
                )
            )

    if full:
        arms.append(
            Arm(
                name="ngram",
                label="+ FlashInfer + n-gram speculation (K=4)",
                kind="opt",
                extra_flags=flashinfer
                + [
                    "--speculative-config",
                    '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}',
                ],
                claim_exact="exact",
                note="Measured: slower (0.82x) and diverged on this hybrid model; kept as a negative result.",
            )
        )
        if mtp_ready:
            arms.append(
                Arm(
                    name="single_seq",
                    label="+ FlashInfer + MTP K=3 + max-num-seqs 1",
                    kind="opt",
                    extra_flags=flashinfer
                    + _mtp_spec(cfg.mtp_head, 3)
                    + ["--max-num-seqs", "1"],
                    claim_exact="exact",
                    note="Single-user scheduling: smaller CUDA graph set, no batching headroom wasted.",
                    requires_mtp=True,
                )
            )
        arms.append(
            Arm(
                name="eager",
                label="Diagnostic: CUDA graphs OFF",
                kind="diagnostic",
                extra_flags=["--enforce-eager"],
                claim_exact="baseline",
                note="Quantifies what CUDA graphs are worth on this GPU.",
            )
        )
    # Run last in every suite: quantifies clock drift and greedy-output noise.
    arms.append(
        Arm(
            name="baseline_recheck",
            label="Baseline re-run (drift / noise floor)",
            kind="opt",
            claim_exact="baseline",
            note="Identical flags to baseline, run last; the quality table uses it as the noise floor.",
        )
    )
    return arms


def select_arms(arms: list[Arm], names: str | None) -> list[Arm]:
    if not names or names == "default":
        return [a for a in arms if a.kind != "diagnostic"]
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if wanted == ["all"]:
        return arms
    by_name = {a.name: a for a in arms}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"unknown arm(s): {', '.join(missing)}; known: {', '.join(by_name)}")
    return [by_name[n] for n in wanted]


def render_commands(cfg: RunConfig, arms: list[Arm]) -> str:
    from .server import build_server_cmd

    lines = []
    for arm in arms:
        cmd = build_server_cmd(
            cfg.model, cfg.served_name, cfg.host, cfg.port, arm.flags(cfg)
        )
        lines.append(f"# --- {arm.name}: {arm.label} ---")
        lines.append(" ".join(_shell_quote(part) for part in cmd))
        lines.append("")
    return "\n".join(lines)


def _shell_quote(part: str) -> str:
    if part and all(c not in part for c in " \t'\"\\$`&|;<>()*?[]{}#!~"):
        return part
    return "'" + part.replace("'", "'\\''") + "'"
