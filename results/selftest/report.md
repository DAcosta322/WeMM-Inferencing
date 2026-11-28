# WeMM-Video-35B-A3B decode TPS benchmark

> **SIMULATED RUN** — these numbers come from the built-in mock server (`--self-test`), not from a GPU. They validate the harness end to end only.

Generated: 2026-11-28T06:16:15+00:00

## Result

- Baseline decode TPS: **158.1 tok/s**
- Best arm: **gptq_tree** at **453.4 tok/s**, **2.87×** vs baseline
- Target of 2.5×: **MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 158.1 | 1.00× | 158.1 | 158.1 | 239 | – | – | reference |
| `baseline_recheck` | ok | 158.0 | 1.00× | 158.1 | 158.0 | 237 | – | 100% |  |
| `gptq` | ok | 231.3 | 1.46× | 231.3 | 231.4 | 239 | – | 100% |  |
| `gptq_mtp3` | ok | 414.4 | 2.62× | 414.2 | 414.6 | 240 | 39% | 100% |  |
| `gptq_tree` | ok | 453.4 | 2.87× | 453.7 | 453.1 | 240 | 16% | 100% |  |
| `mtp3` | ok | 406.3 | 2.57× | 406.5 | 406.0 | 234 | 65% | 100% |  |
| `mtp3_tree` | ok | 320.3 | 2.03× | 320.2 | 320.5 | 235 | 16% | 100% |  |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_recheck` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `gptq` | 1.46× | 1.46× | 1.46× | 1.46× | 1.46× |
| `gptq_mtp3` | 2.62× | 2.62× | 2.62× | 2.62× | 2.62× |
| `gptq_tree` | 2.87× | 2.87× | 2.87× | 2.86× | 2.87× |
| `mtp3` | 2.57× | 2.57× | 2.57× | 2.57× | 2.57× |
| `mtp3_tree` | 2.03× | 2.03× | 2.03× | 2.03× | 2.03× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Speculative decoding

| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted | Tree nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gptq_mtp3` | 39.0% | 2.17 | 472 | 552 | – |
| `gptq_tree` | 15.6% | 3.18 | 322 | 702 | 14 |
| `mtp3` | 65.0% | 2.95 | 347 | 677 | – |
| `mtp3_tree` | 15.6% | 3.18 | 322 | 702 | 14 |

Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. `Acceptance rate` is the raw accepted/drafted ratio (a tree drafts many more tokens per step, so its rate reads low while its mean length is the meaningful number).

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. `First 64 tok` counts prompts whose first 64 generated token ids match exactly (stable even when a late near-tie flips the tail); `Prefix` is the median token-level common prefix over the full output. Read both against the noise-floor row.

Run-to-run noise floor (identical flags, two servers): **100.0%** token prefix. Greedy output is not bit-stable across server restarts on this model, so small differences are not a quality signal.

| Arm | Claim | Exact | First 64 tok | Prefix (median) | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_recheck` | baseline | 4/4 | 4/4 | 100.0% | noise floor |
| `gptq` | likely | 4/4 | 4/4 | 100.0% | identical |
| `gptq_mtp3` | likely | 4/4 | 4/4 | 100.0% | identical |
| `gptq_tree` | likely | 4/4 | 4/4 | 100.0% | identical |
| `mtp3` | exact | 4/4 | 4/4 | 100.0% | identical |
| `mtp3_tree` | exact | 4/4 | 4/4 | 100.0% | identical |

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backends (all logged) | CUDA graphs | Speculative | Quantization | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | – | 1 | auto | 4321 |
| `baseline_recheck` | FLASHINFER | True | – | – | 1 | auto | 4321 |
| `gptq` | FLASHINFER | True | – | gptq_marlin | 1 | auto | 4321 |
| `gptq_mtp3` | FLASHINFER | True | mtp K=3 | gptq_marlin | 1 | auto | 4321 |
| `gptq_tree` | FLASHINFER | True | mtp K=3 tree(14) | gptq_marlin | 1 | auto | 4321 |
| `mtp3` | FLASHINFER | True | mtp K=3 | – | 1 | auto | 4321 |
| `mtp3_tree` | FLASHINFER | True | mtp K=3 tree(14) | – | 1 | auto | 4321 |

## Environment

- `gpu_name`: mock
- `vllm`: mock
- `flashinfer`: mock

## Protocol

- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`.
- TTFT = client time to first generated token; E2E = client time to the final content/finish chunk; output tokens from `usage.completion_tokens`.
- 0 discarded warmups per arm, 1 measured repetitions per prompt, prompts interleaved by repetition to spread thermal drift.
- One server per arm, started fresh and stopped afterwards; prefix caching disabled for every arm.

