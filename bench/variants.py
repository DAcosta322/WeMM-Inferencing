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
    # No explicit --attention-backend: stock vLLM selects TRITON on this GPU and an
    # explicit FLASHINFER pin crashed engine init on the rebuilt environment.
    fp8kv = ["--kv-cache-dtype", "fp8"]

    arms = [
        Arm(
            name="baseline",
            label="Out-of-the-box BF16 (official recipe, TP=1)",
            kind="baseline",
            claim_exact="baseline",
            note="No extra flags; vLLM selects TRITON + CUDA graphs on this GPU.",
        ),
    ]

    mtp_ready = cfg.mtp_head is not None
    if mtp_ready:
        arms.append(
            Arm(
                name="mtp3",
                label="+ MTP K=3 (grafted Qwen head)",
                kind="opt",
                extra_flags=_mtp_spec(cfg.mtp_head, 3),
                claim_exact="exact",
                note="Speculative verification keeps the target distribution; watch the acceptance rate.",
                requires_mtp=True,
            )
        )

    if full:
        arms.append(
            Arm(
                name="fp8kv",
                label="+ FP8 KV cache",
                kind="opt",
                extra_flags=fp8kv,
                claim_exact="likely",
                note="Fails to build its FlashInfer FP8 prefill kernel on this box; run 1 showed +0.1%.",
            )
        )
        arms.append(
            Arm(
                name="fp8kv_scaled",
                label="+ FP8 KV with runtime scales",
                kind="opt",
                extra_flags=fp8kv + ["--calculate-kv-scales"],
                claim_exact="likely",
                note="Same FlashInfer JIT dependency as fp8kv.",
            )
        )
        if mtp_ready:
            for k in (1, 5, 7):
                arms.append(
                    Arm(
                        name=f"mtp{k}",
                        label=f"+ MTP K={k}",
                        kind="opt",
                        extra_flags=_mtp_spec(cfg.mtp_head, k),
                        claim_exact="exact",
                        note="Acceptance-vs-depth sweep.",
                        requires_mtp=True,
                    )
                )
            arms.append(
                Arm(
                    name="mtp3_fp8kv",
                    label="+ MTP K=3 + FP8 KV",
                    kind="opt",
                    extra_flags=fp8kv + _mtp_spec(cfg.mtp_head, 3),
                    claim_exact="likely",
                    note="Shares the FP8 KV JIT failure; run 1's FP8 KV showed no gain.",
                    requires_mtp=True,
                )
            )
            arms.append(
                Arm(
                    name="single_seq",
                    label="+ MTP K=3 + max-num-seqs 1",
                    kind="opt",
                    extra_flags=_mtp_spec(cfg.mtp_head, 3) + ["--max-num-seqs", "1"],
                    claim_exact="exact",
                    note="Single-user scheduling: smaller CUDA graph set.",
                    requires_mtp=True,
                )
            )
        arms.append(
            Arm(
                name="pin_flashinfer",
                label="Diagnostic: explicit --attention-backend FLASHINFER",
                kind="diagnostic",
                extra_flags=["--attention-backend", "FLASHINFER"],
                claim_exact="baseline",
                note="Crashes engine init on the rebuilt env; kept for diagnosis only.",
            )
        )
        arms.append(
            Arm(
                name="ngram",
                label="+ n-gram speculation (K=4)",
                kind="opt",
                extra_flags=[
                    "--speculative-config",
                    '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}',
                ],
                claim_exact="exact",
                note="Measured: 0.82x and diverged on this hybrid model; negative result only.",
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
