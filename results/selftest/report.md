# WeMM-Video-35B-A3B decode TPS benchmark

> **SIMULATED RUN** — these numbers come from the built-in mock server (`--self-test`), not from a GPU. They validate the harness end to end only.

Generated: 2026-11-22T15:21:18+00:00

## Result

- Baseline decode TPS: **45.4 tok/s**
- Best arm: **mtp3_fp8kv** at **83.1 tok/s**, **1.83×** vs baseline
- Target of 2.5×: **NOT MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 45.4 | 1.00× | 45.4 | 45.4 | 315 | – | – | reference |
| `flashinfer` | ok | 47.5 | 1.05× | 47.5 | 47.5 | 296 | – | 100% |  |
| `flashinfer_fp8kv` | ok | 48.5 | 1.07× | 48.5 | 48.5 | 291 | – | 100% |  |
| `mtp3` | ok | 81.7 | 1.80× | 81.7 | 81.8 | 307 | 72% | 100% |  |
| `mtp3_fp8kv` | ok | 83.1 | 1.83× | 83.1 | 83.1 | 301 | 72% | 100% |  |
| `ngram` | ok | 49.9 | 1.10× | 49.9 | 49.9 | 296 | 25% | 100% |  |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flashinfer` | 1.05× | 1.05× | 1.05× | 1.05× | 1.05× |
| `flashinfer_fp8kv` | 1.07× | 1.07× | 1.07× | 1.07× | 1.07× |
| `mtp3` | 1.80× | 1.80× | 1.80× | 1.80× | 1.80× |
| `mtp3_fp8kv` | 1.83× | 1.83× | 1.83× | 1.83× | 1.83× |
| `ngram` | 1.10× | 1.10× | 1.10× | 1.10× | 1.10× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Speculative decoding

| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted |
| --- | ---: | ---: | ---: | ---: |
| `mtp3` | 72.0% | 3.16 | 324 | 700 |
| `mtp3_fp8kv` | 72.0% | 3.16 | 324 | 700 |
| `ngram` | 25.0% | 2.00 | 512 | 512 |

Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. A K=3 run at 70% acceptance approaches ~2.0× over the same non-speculative server.

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. Token-level prefix ratio is computed with the server tokenizer; 100% means the two outputs are token-identical.

| Arm | Claim | Exact matches | Token prefix (median) | Verdict |
| --- | --- | ---: | ---: | --- |
| `flashinfer` | exact | 4/4 | 100.0% | OK (identical) |
| `flashinfer_fp8kv` | likely | 4/4 | 100.0% | exact |
| `mtp3` | exact | 4/4 | 100.0% | OK (identical) |
| `mtp3_fp8kv` | likely | 4/4 | 100.0% | exact |
| `ngram` | exact | 4/4 | 100.0% | OK (identical) |

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backend | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | 1 | auto | 4321 |
| `flashinfer` | FLASHINFER | True | – | 1 | auto | 4321 |
| `flashinfer_fp8kv` | FLASHINFER | True | – | 1 | fp8 | 4321 |
| `mtp3` | FLASHINFER | True | mtp K=3 | 1 | auto | 4321 |
| `mtp3_fp8kv` | FLASHINFER | True | mtp K=3 | 1 | fp8 | 4321 |
| `ngram` | FLASHINFER | True | ngram K=4 | 1 | auto | 4321 |

## Environment

- `gpu_name`: mock
- `vllm`: mock
- `flashinfer`: mock

## Protocol

- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`.
- TTFT = client time to first generated token; E2E = client time to the final content/finish chunk; output tokens from `usage.completion_tokens`.
- 0 discarded warmups per arm, 1 measured repetitions per prompt, prompts interleaved by repetition to spread thermal drift.
- One server per arm, started fresh and stopped afterwards; prefix caching disabled for every arm.

