# WeMM-Video-35B-A3B decode TPS benchmark

> **SIMULATED RUN** — these numbers come from the built-in mock server (`--self-test`), not from a GPU. They validate the harness end to end only.

Generated: 2026-11-27T07:03:21+00:00

## Result

- Baseline decode TPS: **45.4 tok/s**
- Best arm: **mtp3_fp8kv** at **83.2 tok/s**, **1.83×** vs baseline
- Target of 2.5×: **NOT MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 45.4 | 1.00× | 45.4 | 45.4 | 312 | – | – | reference |
| `baseline_recheck` | ok | 45.4 | 1.00× | 45.4 | 45.4 | 312 | – | 100% |  |
| `flashinfer` | ok | 47.6 | 1.05× | 47.6 | 47.6 | 293 | – | 100% |  |
| `flashinfer_fp8kv` | ok | 48.5 | 1.07× | 48.5 | 48.5 | 287 | – | 100% |  |
| `mtp3` | ok | 81.8 | 1.80× | 81.8 | 81.8 | 303 | 72% | 100% |  |
| `mtp3_fp8kv` | ok | 83.2 | 1.83× | 83.2 | 83.2 | 298 | 72% | 100% |  |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_recheck` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |
| `flashinfer` | 1.05× | 1.05× | 1.05× | 1.05× | 1.05× |
| `flashinfer_fp8kv` | 1.07× | 1.07× | 1.07× | 1.07× | 1.07× |
| `mtp3` | 1.80× | 1.80× | 1.80× | 1.80× | 1.80× |
| `mtp3_fp8kv` | 1.83× | 1.83× | 1.83× | 1.83× | 1.83× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Speculative decoding

| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted |
| --- | ---: | ---: | ---: | ---: |
| `mtp3` | 72.0% | 3.16 | 324 | 700 |
| `mtp3_fp8kv` | 72.0% | 3.16 | 324 | 700 |

Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. A K=3 run at 70% acceptance approaches ~2.0× over the same non-speculative server.

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. Token-level prefix ratio is computed with the server tokenizer; 100% means the two outputs are token-identical.

Run-to-run noise floor (identical flags, two servers): **100.0%** token prefix. Greedy output is not bit-stable across server restarts on this model, so small differences are not a quality signal.

| Arm | Claim | Exact matches | Token prefix (median) | Verdict |
| --- | --- | ---: | ---: | --- |
| `baseline_recheck` | baseline | 4/4 | 100.0% | noise floor |
| `flashinfer` | exact | 4/4 | 100.0% | identical |
| `flashinfer_fp8kv` | likely | 4/4 | 100.0% | identical |
| `mtp3` | exact | 4/4 | 100.0% | identical |
| `mtp3_fp8kv` | likely | 4/4 | 100.0% | identical |

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backend | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | 1 | auto | 4321 |
| `baseline_recheck` | FLASHINFER | True | – | 1 | auto | 4321 |
| `flashinfer` | FLASHINFER | True | – | 1 | auto | 4321 |
| `flashinfer_fp8kv` | FLASHINFER | True | – | 1 | fp8 | 4321 |
| `mtp3` | FLASHINFER | True | mtp K=3 | 1 | auto | 4321 |
| `mtp3_fp8kv` | FLASHINFER | True | mtp K=3 | 1 | fp8 | 4321 |

## Environment

- `gpu_name`: mock
- `vllm`: mock
- `flashinfer`: mock

## Protocol

- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`.
- TTFT = client time to first generated token; E2E = client time to the final content/finish chunk; output tokens from `usage.completion_tokens`.
- 0 discarded warmups per arm, 1 measured repetitions per prompt, prompts interleaved by repetition to spread thermal drift.
- One server per arm, started fresh and stopped afterwards; prefix caching disabled for every arm.

