"""Deterministic prompt suite.

Decode TPS is normalised for TTFT, so the suite focuses on generating a fixed
number of output tokens at two context lengths:

  * ``short``  ~40-80 prompt tokens     (chat / quick question)
  * ``long``   ~3.5-4.5k prompt tokens  (video-length context, text only)

Every prompt is unique so that prefix caching (disabled anyway) cannot
influence the measurement, and every prompt is greedy-decoded with
``ignore_eos`` so the output length is identical across arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prompt:
    id: str
    bucket: str  # "short" | "long"
    text: str
    kind: str = "bench"  # "bench" | "quality"
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Short bench prompts
# ---------------------------------------------------------------------------

SHORT_TEXTS = [
    (
        "You are explaining a technical topic to an engineering team. "
        "Describe in detail how a mixture-of-experts transformer routes tokens "
        "to experts, why load balancing matters, and what can go wrong when the "
        "router collapses onto a small number of experts. Give concrete examples."
    ),
    (
        "A team is planning a video understanding pipeline that accepts frames "
        "and audio tracks. Walk through the full data path from decoding a clip "
        "to producing a grounded textual answer, including frame sampling, "
        "timestamp tokens, audio encoding, and how a language model consumes the "
        "result. Be specific about the engineering tradeoffs at each stage."
    ),
]

# ---------------------------------------------------------------------------
# Long-context bench prompts
# ---------------------------------------------------------------------------
# Six paragraphs, cycled, so the prompt is long, coherent, and has no exact
# duplicated sentences back-to-back (which could inflate n-gram acceptance in
# an unfair way if a speculative method ever uses prompt lookup).

_LONG_PARAGRAPHS = [
    (
        "Section 1: Operating a shared inference cluster. Capacity planning starts by "
        "separating prefill work from decode work, because the two have different "
        "bottlenecks. Prefill is compute bound and scales with the number of prompt "
        "tokens multiplied by the model width. Decode is latency bound at small batch "
        "sizes and is dominated by per-token weight traffic and kernel launch overhead. "
        "A cluster that mixes long prompts with many short chats will therefore show "
        "bimodal latency, and the scheduler has to decide whether to chunk a long "
        "prefill or to admit more decode steps first. Operators usually pin a maximum "
        "batched token count so that one very long request cannot stall every other "
        "request. The tradeoff is that chunking a prefill adds first-token latency for "
        "that request in exchange for steadier inter-token latency elsewhere."
    ),
    (
        "Section 2: Quantisation for inference. Weight-only quantisation reduces the "
        "bytes that must be read for each generated token, which is the main lever for "
        "single-stream decode throughput on memory-bandwidth-limited devices. "
        "Activation quantisation reduces arithmetic cost but requires careful "
        "calibration because outliers in a few channels can dominate the error. "
        "Group-wise scales, typically one scale per 128 weights, recover most of the "
        "accuracy lost at four bits. The practical question is never only about "
        "perplexity: a quantised model must also keep instruction following, refusal "
        "behaviour and long-context retrieval intact. Teams therefore keep a small "
        "regression suite of prompts and compare outputs against the full-precision "
        "checkpoint, treating any change in the answer as a review item rather than "
        "an automatic failure."
    ),
    (
        "Section 3: Speculative decoding. A speculative method proposes several future "
        "tokens with a cheap model, then verifies them in a single forward pass of the "
        "target model. Because verification keeps only tokens that the target would "
        "have produced under its own distribution, greedy output is unchanged and "
        "sampled output remains statistically identical. The speedup is bounded by two "
        "quantities: the acceptance rate of the proposals and the cost of the proposal "
        "step itself. Multi-token prediction heads are attractive because they live "
        "inside the model checkpoint and add only one small decoder layer, so the "
        "proposal cost is a few percent of a full forward pass. The main risk is "
        "distribution shift: if the head was trained beside a base model and is later "
        "paired with a fine-tuned or post-trained model, acceptance can fall and the "
        "expected speedup with it."
    ),
    (
        "Section 4: Attention variants. Full attention keeps a per-token key and value "
        "cache, so its memory footprint grows linearly with context length. Linear "
        "attention variants replace the growing cache with a fixed-size recurrent "
        "state, trading exact retrieval for constant memory. Hybrid stacks interleave "
        "the two, for example three linear layers followed by one full-attention layer, "
        "to keep most of the efficiency while retaining a mechanism for long-range "
        "lookup. Serving such a model requires two different cache layouts in the same "
        "engine, and the scheduler must allocate blocks for both. A common mistake is "
        "to tune the full-attention cache dtype while forgetting that the recurrent "
        "state has its own precision requirement, which is often stricter."
    ),
    (
        "Section 5: Measurement discipline. A latency number is only useful if the "
        "protocol that produced it is documented. Report the prompt token count, the "
        "generated token count, whether the run was warm, how many repetitions were "
        "used, and how outliers were handled. Separate first-token latency from "
        "inter-token latency, because a single end-to-end average hides the fact that "
        "a long prompt can dominate the wall clock while decode speed is unchanged. "
        "Pin clocks or record them, since a boost clock at the start of a run can make "
        "the first measurement look better than the steady state. Finally, always "
        "report the median and a spread alongside any speedup claim; a ratio computed "
        "from two noisy single runs is not evidence."
    ),
    (
        "Section 6: Reliability of multimodal serving. Media inputs arrive in "
        "inconsistent formats, with variable frame rates, missing audio tracks, "
        "rotation metadata and codecs that decode differently on different hosts. A "
        "serving stack should normalise inputs before they reach the model, cache "
        "decoded frames by content hash, and cap the number of visual tokens per "
        "request so that one oversized clip cannot exhaust the context window. "
        "Observability matters as much as throughput: log the number of frames, the "
        "number of audio tokens and the prompt length for every request, and treat a "
        "sudden change in those distributions as a signal that an upstream client "
        "changed its behaviour. Without that, a throughput regression is easily "
        "misattributed to the model or the accelerator."
    ),
]


def _long_prompt(cycles: int, question: str) -> str:
    body = "\n\n".join(
        _LONG_PARAGRAPHS[i % len(_LONG_PARAGRAPHS)] for i in range(len(_LONG_PARAGRAPHS) * cycles)
    )
    return (
        "Read the following operational notes carefully.\n\n"
        f"{body}\n\n"
        f"{question}"
    )


LONG_TEXTS = [
    _long_prompt(
        4,
        "Question: Based only on the notes above, explain which two quantities bound "
        "the speedup of speculative decoding, and why a hybrid attention stack needs "
        "two cache layouts. Answer in a structured paragraph.",
    ),
    _long_prompt(
        4,
        "Question: Using the notes above, propose a measurement protocol for a "
        "single-user decode benchmark. State exactly what to control, what to report, "
        "and how to decide whether an optimisation is real.",
    ),
]


# ---------------------------------------------------------------------------
# Quality prompts (free-form, not length forced)
# ---------------------------------------------------------------------------

QUALITY_PROMPTS = [
    Prompt(
        id="q_math",
        bucket="short",
        kind="quality",
        text=(
            "A meeting starts at 09:45 and lasts 2 hours and 35 minutes. "
            "What time does it end? Give the answer and the arithmetic."
        ),
    ),
    Prompt(
        id="q_factual",
        bucket="short",
        kind="quality",
        text=(
            "Name the three primary colours of light, state what happens when all "
            "three are combined at equal intensity, and give one practical use of "
            "that fact."
        ),
    ),
    Prompt(
        id="q_reasoning",
        bucket="short",
        kind="quality",
        text=(
            "A train leaves at 14:10 and arrives at 17:05 after travelling 245 km. "
            "Compute the average speed in km/h, show your working, and check whether "
            "the result is plausible for a regional service."
        ),
    ),
    Prompt(
        id="q_longctx",
        bucket="long",
        kind="quality",
        text=_long_prompt(
            2,
            "Question: List, in order, the six section topics of the notes above, "
            "then state one specific risk mentioned in the section about speculative "
            "decoding.",
        ),
    ),
]


def bench_suite() -> list[Prompt]:
    """The measured suite, ordered to interleave context lengths."""
    shorts = [
        Prompt(id=f"s{idx + 1}", bucket="short", text=text)
        for idx, text in enumerate(SHORT_TEXTS)
    ]
    longs = [
        Prompt(id=f"l{idx + 1}", bucket="long", text=text)
        for idx, text in enumerate(LONG_TEXTS)
    ]
    # Interleave context lengths so thermal drift hits both buckets equally.
    prompts: list[Prompt] = []
    for idx in range(max(len(shorts), len(longs))):
        if idx < len(shorts):
            prompts.append(shorts[idx])
        if idx < len(longs):
            prompts.append(longs[idx])
    return prompts


def quality_suite() -> list[Prompt]:
    return list(QUALITY_PROMPTS)


def suite_fingerprint(prompts: list[Prompt]) -> str:
    from .common import stable_hash

    return stable_hash("\n\n".join(p.text for p in prompts), 16)
