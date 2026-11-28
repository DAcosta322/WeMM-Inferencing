"""Arm definitions: exactly which server flags each measurement runs with.

Arms are deliberately small and additive so that the contribution of each
optimisation is visible:

    baseline          official WeMM recipe, BF16, TP=1, nothing else
    <quant>           + --quantization <method> on a quantised checkpoint
    mtp3              + Qwen3.5 MTP head, K=3
    mtp3_tree         + MTP K=3 with a depth-3 / branch-2 speculation tree
    <quant>_mtp3      quantised weights + MTP K=3
    <quant>_tree      quantised weights + MTP K=3 tree
    flashinfer*       + --attention-backend FLASHINFER  (* --full, diagnostic)
    fp8kv*            + --kv-cache-dtype fp8            (* --full)
    mtp3_fp8kv*       + MTP K=3 + FP8 KV                (* --full)
    mtp5*/mtp1*/mtp7* MTP K sweep                       (* --full)
    ngram*            n-gram speculation                (* --full, negative result)
    eager*            CUDA graphs off                   (* --full, diagnostic only)

``<quant>`` is the selected weight-quantisation method (``--quant-method``,
default ``awq``). Every arm runs the same prompts, seeds, token limits and
measurement protocol. Speculative arms are lossless for greedy output (the
target re-verifies every draft token); quantised arms are lossy and are marked
``claim_exact="likely"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


TREE_DEPTH = 3
TREE_BRANCHING = 2

# ``--moe-backend`` values accepted by vLLM 0.18.0 (vllm/config/kernel.py
# ``MoEBackend`` Literal). Normalisation lower-cases and turns dashes into
# underscores, so these are the canonical spellings.
MOE_BACKENDS = (
    "auto",
    "triton",
    "deep_gemm",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
)

# Unquantised BF16 MoE supports only this subset (the rest are for quantised
# expert kernels); see vllm/.../fused_moe/oracle/unquantized.py.
MOE_BACKENDS_UNQUANTIZED = ("triton", "flashinfer_trtllm", "flashinfer_cutlass")

# ``--optimization-level`` / ``-O`` values (0 = none … 3 = full).
OPT_LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class QuantMethod:
    """A weight-quantisation method the harness can benchmark."""

    name: str
    flag: str  # vLLM --quantization value
    label: str
    bits: int
    note: str


# Ordered so ``awq`` (the recommended default) is first. The vLLM ``flag`` values
# are verified against vllm/model_executor/layers/quantization/__init__.py (0.18.0).
QUANT_METHODS: dict[str, QuantMethod] = {
    "awq": QuantMethod(
        "awq", "awq_marlin", "AWQ int4 (W4A16)", 4,
        "Equal-or-better than GPTQ with simpler calibration; a WeMM AWQ checkpoint "
        "already exists.",
    ),
    "nvfp4": QuantMethod(
        "nvfp4", "modelopt_mixed", "NVFP4 (W4A16, Blackwell)", 4,
        "Tencent's -NVFP4 checkpoint is ModelOpt MIXED_PRECISION (NVFP4 experts, BF16 "
        "elsewhere); needs the modelopt_mixed vLLM method plus the W4A16_NVFP4 backport.",
    ),
    "gptq": QuantMethod(
        "gptq", "gptq_marlin", "GPTQ int4 (W4A16)", 4,
        "Reference int4 method; activation-aware Hessian, calibration-sensitive.",
    ),
    "fp8": QuantMethod(
        "fp8", "fp8", "FP8 (W8A8)", 8,
        "Safer for the recurrent GDN/linear-attention layers (error accumulates across "
        "tokens); halves weight bytes rather than quartering them.",
    ),
}


def quant_flags(method: str) -> list[str]:
    return ["--quantization", QUANT_METHODS[method].flag]


def moe_flags(backend: str) -> list[str]:
    return ["--moe-backend", backend]


def opt_level_flags(level: int) -> list[str]:
    return ["--optimization-level", str(int(level))]


def compilation_flags(config_json: str) -> list[str]:
    return ["--compilation-config", config_json]


@dataclass
class RunConfig:
    model: str = "tencent/WeMM-Video-35BA3B"
    quant_model: str | None = None  # path to a quantised checkpoint (method via quant_method)
    quant_method: str = "awq"  # key into QUANT_METHODS
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
    moe_backend: str | None = None  # --moe-backend: single-arm MoE kernel selection
    opt_level: int | None = None  # --opt-level / -O: single-arm optimisation level
    compilation_config: str | None = None  # --compilation-config JSON: single-arm
    prefix_caching: bool = False  # --prefix-caching: add the prefix_cache arm

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


def arm_spec_config(arm: "Arm") -> dict | None:
    """Return the parsed ``--speculative-config`` JSON for an arm, if any."""
    flags = arm.extra_flags
    for i, flag in enumerate(flags):
        if flag == "--speculative-config" and i + 1 < len(flags):
            try:
                return json.loads(flags[i + 1])
            except (ValueError, TypeError):
                return None
    return None


def arm_quantization(arm: "Arm") -> str | None:
    """Return the ``--quantization`` value for an arm, if any."""
    flags = arm.extra_flags
    for i, flag in enumerate(flags):
        if flag == "--quantization" and i + 1 < len(flags):
            return flags[i + 1]
    return None


@dataclass
class Arm:
    name: str
    label: str
    kind: str  # baseline | opt | diagnostic
    extra_flags: list[str] = field(default_factory=list)
    claim_exact: str = "exact"  # exact | likely | no | baseline | unknown
    note: str = ""
    requires_mtp: bool = False
    requires_quant: bool = False
    model: str | None = None  # override RunConfig.model (e.g. a quantised checkpoint)
    prefix_cache: bool = False  # --enable-prefix-caching + fixed system prompt

    def flags(self, cfg: RunConfig) -> list[str]:
        flags = cfg.common_flags()
        if self.prefix_cache:
            # The default suite disables prefix caching for every arm; this arm
            # turns it back on and, together with a fixed system prompt, makes
            # the shared prefix actually cacheable.
            flags = [f for f in flags if f != "--no-enable-prefix-caching"]
            flags.append("--enable-prefix-caching")
        return flags + self.extra_flags

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "model": self.model,
            "extra_flags": self.extra_flags,
            "claim_exact": self.claim_exact,
            "note": self.note,
            "requires_mtp": self.requires_mtp,
            "requires_quant": self.requires_quant,
            "prefix_cache": self.prefix_cache,
            "speculative": arm_spec_config(self),
            "quantization": arm_quantization(self),
        }


def tree_choices(
    depth: int = TREE_DEPTH, branching: int = TREE_BRANCHING
) -> list[tuple[int, ...]]:
    """Breadth-first uniform-branching speculation tree.

    Each node is a tuple of child indices describing its path from the root
    (``()``): ``(0,)`` is the first root child, ``(0, 1)`` its second child.
    vLLM re-sorts this breadth-first anyway; we emit it pre-sorted.
    """
    nodes: list[tuple[int, ...]] = []
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(depth):
        nxt: list[tuple[int, ...]] = []
        for parent in frontier:
            for i in range(branching):
                node = parent + (i,)
                nodes.append(node)
                nxt.append(node)
        frontier = nxt
    nodes.sort(key=lambda t: (len(t), t))
    return nodes


def tree_spec_string(
    depth: int = TREE_DEPTH, branching: int = TREE_BRANCHING
) -> str:
    """The ``speculative_token_tree`` literal vLLM expects (a tuple path list)."""
    return repr(tree_choices(depth, branching))


def tree_node_count(depth: int = TREE_DEPTH, branching: int = TREE_BRANCHING) -> int:
    return sum(branching**i for i in range(1, depth + 1))


def _mtp_spec(head: Path | None, k: int) -> list[str]:
    if head is None:
        raise ValueError("MTP arm requested but no --mtp-head path is set")
    payload = {
        "model": str(head),
        "method": "mtp",
        "num_speculative_tokens": k,
    }
    return ["--speculative-config", json.dumps(payload, separators=(",", ":"))]


def _mtp_tree_spec(
    head: Path | None,
    k: int = TREE_DEPTH,
    branching: int = TREE_BRANCHING,
) -> list[str]:
    if head is None:
        raise ValueError("MTP tree arm requested but no --mtp-head path is set")
    payload = {
        "model": str(head),
        "method": "mtp",
        "num_speculative_tokens": k,
        "speculative_token_tree": tree_spec_string(k, branching),
    }
    return ["--speculative-config", json.dumps(payload, separators=(",", ":"))]


def build_arms(cfg: RunConfig, full: bool = False) -> list[Arm]:
    # No explicit --attention-backend: stock vLLM selects TRITON on this GPU and an
    # explicit FLASHINFER pin crashed engine init on the rebuilt environment.
    fp8kv = ["--kv-cache-dtype", "fp8"]

    arms = [
        Arm(
            name="baseline",
            label=f"Out-of-the-box BF16 (official recipe, TP={cfg.tp_size})",
            kind="baseline",
            claim_exact="baseline",
            note="No extra flags; vLLM selects TRITON + CUDA graphs on this GPU.",
        ),
    ]

    mtp_ready = cfg.mtp_head is not None
    qname = cfg.quant_method
    if qname not in QUANT_METHODS:
        raise ValueError(
            f"unknown --quant-method {qname!r}; choose from {', '.join(QUANT_METHODS)}"
        )
    qm = QUANT_METHODS[qname]

    # Quantised arms are always built (they need --quant-model to actually run);
    # the orchestrator drops or errors on them when the checkpoint is missing.
    arms.append(
        Arm(
            name=qname,
            label=qm.label,
            kind="opt",
            model=cfg.quant_model,
            extra_flags=quant_flags(qname),
            claim_exact="likely",
            note=f"{qm.note} Lossy: relaxes the token-exact gate.",
            requires_quant=True,
        )
    )

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
        arms.append(
            Arm(
                name="mtp3_tree",
                label=f"+ MTP K=3 tree (depth {TREE_DEPTH}, branch {TREE_BRANCHING})",
                kind="opt",
                extra_flags=_mtp_tree_spec(cfg.mtp_head, 3),
                claim_exact="exact",
                note=(
                    f"Single-forward tree verification over {tree_node_count()} nodes; "
                    "hedges near-ties, but verify cost can offset the extra acceptance."
                ),
                requires_mtp=True,
            )
        )
        arms.append(
            Arm(
                name=f"{qname}_mtp3",
                label=f"{qm.label} + MTP K=3",
                kind="opt",
                model=cfg.quant_model,
                extra_flags=quant_flags(qname) + _mtp_spec(cfg.mtp_head, 3),
                claim_exact="likely",
                note="Quantised target drifts the draft head's argmax; expect lower acceptance (p~0.6).",
                requires_mtp=True,
                requires_quant=True,
            )
        )
        arms.append(
            Arm(
                name=f"{qname}_tree",
                label=f"{qm.label} + MTP tree (depth {TREE_DEPTH}, branch {TREE_BRANCHING})",
                kind="opt",
                model=cfg.quant_model,
                extra_flags=quant_flags(qname) + _mtp_tree_spec(cfg.mtp_head, 3),
                claim_exact="likely",
                note="Quantised target + tree verification; highest-yield combination.",
                requires_mtp=True,
                requires_quant=True,
            )
        )

    # -- opt-in optimisation-idea arms (3/4/6) ---------------------------
    # One arm per explicitly requested knob, additive on top of the BF16
    # baseline so each knob's contribution is isolated.
    if cfg.moe_backend:
        arms.append(
            Arm(
                name=f"moe_{cfg.moe_backend}",
                label=f"MoE backend: {cfg.moe_backend}",
                kind="opt",
                extra_flags=moe_flags(cfg.moe_backend),
                claim_exact="exact",
                note=(
                    "Explicit MoE expert kernel selection; CUTLASS/FlashInfer "
                    "kernels can beat Triton on Blackwell — measure."
                ),
            )
        )
    if cfg.opt_level is not None:
        level = int(cfg.opt_level)
        arms.append(
            Arm(
                name=f"o{level}",
                label=f"Optimisation level -O{level}",
                kind="opt",
                extra_flags=opt_level_flags(level),
                claim_exact="exact",
                note=(
                    "vLLM -O level: 0 = none, 1 = Dynamo+piecewise graphs, "
                    "2 = + full graphs (default), 3 = same as 2 today."
                ),
            )
        )
    if cfg.compilation_config:
        arms.append(
            Arm(
                name="compile_cfg",
                label="Custom compilation config",
                kind="opt",
                extra_flags=compilation_flags(cfg.compilation_config),
                claim_exact="exact",
                note=(
                    "--compilation-config as supplied; the hybrid GDN stack may "
                    "only partially compile."
                ),
            )
        )
    if cfg.prefix_caching:
        arms.append(
            Arm(
                name="prefix_cache",
                label="Prefix caching + fixed system prompt",
                kind="opt",
                prefix_cache=True,
                claim_exact="no",
                note=(
                    "--enable-prefix-caching plus a fixed system prompt shared "
                    "across requests; cuts TTFT in production, does not change "
                    "decode TPS, and the added system prompt means output is not "
                    "comparable to baseline."
                ),
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
            for k in (5, 7):
                arms.append(
                    Arm(
                        name=f"mtp{k}_tree",
                        label=f"+ MTP K={k} tree (depth {k}, branch {TREE_BRANCHING})",
                        kind="opt",
                        extra_flags=_mtp_tree_spec(cfg.mtp_head, k),
                        claim_exact="exact",
                        note=(
                            f"{tree_node_count(k, TREE_BRANCHING)}-node tree "
                            "verification; deeper than the default K=3 tree, so "
                            "watch whether verify cost offsets the extra acceptance."
                        ),
                        requires_mtp=True,
                    )
                )
        arms.append(
            Arm(
                name="pin_flashinfer",
                label="Diagnostic: explicit --attention-backend FLASHINFER",
                kind="diagnostic",
                extra_flags=["--attention-backend", "FLASHINFER"],
                claim_exact="likely",
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
                claim_exact="exact",
                note="Quantifies what CUDA graphs are worth on this GPU.",
            )
        )
        # idea 3: explicit MoE expert backends for unquantised BF16.
        for backend in MOE_BACKENDS_UNQUANTIZED:
            arms.append(
                Arm(
                    name=f"moe_{backend}",
                    label=f"MoE backend: {backend}",
                    kind="opt",
                    extra_flags=moe_flags(backend),
                    claim_exact="exact",
                    note=(
                        "FlashInfer TRTLLM/CUTLASS kernels can beat Triton on "
                        "Blackwell; they need FlashInfer and may fail — measure."
                        if backend != "triton"
                        else "Explicit Triton MoE control (the default)."
                    ),
                )
            )
        # idea 4: optimisation-level extremes (baseline uses the default -O2).
        arms.append(
            Arm(
                name="o3",
                label="Optimisation level -O3",
                kind="opt",
                extra_flags=opt_level_flags(3),
                claim_exact="exact",
                note="-O3 currently aliases -O2 in vLLM 0.18.0; a forward-looking check.",
            )
        )
        arms.append(
            Arm(
                name="o0",
                label="Optimisation level -O0",
                kind="diagnostic",
                extra_flags=opt_level_flags(0),
                claim_exact="exact",
                note="No compilation, no CUDA graphs; isolates what the -O stack contributes.",
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
    # Deduplicate by name: e.g. --moe-backend triton plus --full's MoE sweep, or
    # --opt-level 3 plus --full's -O3 arm, must not run the same arm twice.
    seen: set[str] = set()
    unique: list[Arm] = []
    for arm in arms:
        if arm.name in seen:
            continue
        seen.add(arm.name)
        unique.append(arm)
    return unique


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
        model = arm.model or cfg.model
        if arm.requires_quant and not arm.model:
            model = "<QUANT-CHECKPOINT>"
        cmd = build_server_cmd(
            model, cfg.served_name, cfg.host, cfg.port, arm.flags(cfg)
        )
        lines.append(f"# --- {arm.name}: {arm.label} ---")
        lines.append(" ".join(_shell_quote(part) for part in cmd))
        lines.append("")
    return "\n".join(lines)


def _shell_quote(part: str) -> str:
    if part and all(c not in part for c in " \t'\"\\$`&|;<>()*?[]{}#!~"):
        return part
    return "'" + part.replace("'", "'\\''") + "'"
