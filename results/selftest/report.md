# WeMM-Video-35B-A3B decode TPS benchmark

> **SIMULATED RUN** — these numbers come from the built-in mock server (`--self-test`), not from a GPU. They validate the harness end to end only.

Generated: 2026-11-28T11:22:08+00:00

## Result

- Baseline decode TPS: **158.1 tok/s**
- Best arm: **awq_tree** at **494.5 tok/s**, **3.13×** vs baseline
- Target of 2.5×: **MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 158.1 | 1.00× | 158.2 | 158.1 | 237 | – | – | reference |
| `awq` | ok | 236.8 | 1.50× | 236.8 | 236.8 | 239 | – | 100% |  |
| `awq_mtp3` | ok | 423.5 | 2.68× | 423.5 | 423.3 | 239 | 39% | 100% |  |
| `awq_tree` | ok | 494.5 | 3.13× | 493.6 | 495.3 | 240 | 16% | 100% |  |
| `baseline_recheck` | ok | 158.1 | 1.00× | 158.1 | 158.1 | 237 | – | 100% |  |
| `mtp3` | ok | 406.1 | 2.57× | 406.1 | 406.0 | 236 | 65% | 100% |  |
| `mtp3_tree` | ok | 320.0 | 2.02× | 320.0 | 320.2 | 235 | 16% | 100% |  |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `awq` | 1.50× | 1.50× | 1.50× | 1.50× | 1.50× |
| `awq_mtp3` | 2.68× | 2.68× | 2.68× | 2.68× | 2.68× |
| `awq_tree` | 3.12× | 3.13× | 3.12× | 3.13× | 3.13× |
| `baseline_recheck` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `mtp3` | 2.57× | 2.57× | 2.56× | 2.56× | 2.57× |
| `mtp3_tree` | 2.02× | 2.03× | 2.02× | 2.02× | 2.02× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Speculative decoding

| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted | Tree nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `awq_mtp3` | 39.0% | 2.17 | 472 | 552 | – |
| `awq_tree` | 15.6% | 3.18 | 322 | 702 | 14 |
| `mtp3` | 65.0% | 2.95 | 347 | 677 | – |
| `mtp3_tree` | 15.6% | 3.18 | 322 | 702 | 14 |

Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. `Acceptance rate` is the raw accepted/drafted ratio (a tree drafts many more tokens per step, so its rate reads low while its mean length is the meaningful number).

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. `First 64 tok` counts prompts whose first 64 generated token ids match exactly (stable even when a late near-tie flips the tail); `Prefix` is the median token-level common prefix over the full output. Read both against the noise-floor row.

Run-to-run noise floor (identical flags, two servers): **100.0%** token prefix. Greedy output is not bit-stable across server restarts on this model, so small differences are not a quality signal.

| Arm | Claim | Exact | First 64 tok | Prefix (median) | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `awq` | likely | 4/4 | 4/4 | 100.0% | identical |
| `awq_mtp3` | likely | 4/4 | 4/4 | 100.0% | identical |
| `awq_tree` | likely | 4/4 | 4/4 | 100.0% | identical |
| `baseline_recheck` | baseline | 4/4 | 4/4 | 100.0% | noise floor |
| `mtp3` | exact | 4/4 | 4/4 | 100.0% | identical |
| `mtp3_tree` | exact | 4/4 | 4/4 | 100.0% | identical |

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backends (all logged) | CUDA graphs | Speculative | Quantization | MoE backend | Opt | Prefix cache | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | – | – | default | – | 1 | auto | 4321 |
| `awq` | FLASHINFER | True | – | awq_marlin | – | default | – | 1 | auto | 4321 |
| `awq_mtp3` | FLASHINFER | True | mtp K=3 | awq_marlin | – | default | – | 1 | auto | 4321 |
| `awq_tree` | FLASHINFER | True | mtp K=3 tree(14) | awq_marlin | – | default | – | 1 | auto | 4321 |
| `baseline_recheck` | FLASHINFER | True | – | – | – | default | – | 1 | auto | 4321 |
| `mtp3` | FLASHINFER | True | mtp K=3 | – | – | default | – | 1 | auto | 4321 |
| `mtp3_tree` | FLASHINFER | True | mtp K=3 tree(14) | – | – | default | – | 1 | auto | 4321 |

## Environment

- `gpu_name`: mock
- `vllm`: mock
- `flashinfer`: mock

## Protocol

- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`.
- TTFT = client time to first generated token; E2E = client time to the final content/finish chunk; output tokens from `usage.completion_tokens`.
- 0 discarded warmups per arm, 1 measured repetitions per prompt, prompts interleaved by repetition to spread thermal drift.
- One server per arm, started fresh and stopped afterwards; prefix caching is disabled for every arm except `prefix_cache`, which enables it together with a fixed system prompt.

