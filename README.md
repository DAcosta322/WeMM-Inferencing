# WeMM-Video-35B-A3B — single-GPU decode-TPS optimisation harness

`idea.md` is unchanged. Everything I produced lives here, and this file is the write-up:
findings, what the harness does, how to run it on the RTX Pro 6000 Blackwell, and what to do
with the numbers.

**Status:** harness complete, unit-tested, and validated end-to-end against a mock server.
The real measurements have not been run yet — this machine has no working GPU
(`nvidia-smi` fails, Python 3.14), so every number below is either arithmetic or explicitly
marked as simulated.

---

## TL;DR

* **The MTP head is missing from every released WeMM checkpoint.** `mtp_num_hidden_layers: 1`
  is in the config, but the BF16 checkpoint has 0/32,211 and the AWQ checkpoint 0/93,891
  `mtp.*` tensors. `--speculative-config '{"method":"mtp"}'` cannot work out of the box.
* **The fix is a graft, not a retrain.** Upstream `Qwen/Qwen3.5-35B-A3B` (same 40-layer,
  256-expert, 2048-hidden backbone) ships the full 785-tensor MTP head. The harness builds a
  ~3.7 GB sidecar draft directory from those `mtp.*` tensors plus the WeMM `lm_head` and
  `embed_tokens`, and points vLLM at it. Speculative decoding verifies every draft token, so
  the model's output is unchanged — only the acceptance rate (and the speedup) depends on the
  graft.
* **CUDA graphs are already on by default** (`--enforce-eager` defaults to false), so that
  part of the brief is a no-op unless something disabled it. The harness confirms it from the
  server log rather than trusting the flags, and includes an `eager` diagnostic arm.
* **FlashInfer is probably *not* what vLLM picks on this GPU.** vLLM 0.18 prefers
  `FLASH_ATTN` for compute capability major ≠ 10; the RTX Pro 6000 Blackwell is sm_120
  (major 12). Pinning `--attention-backend FLASHINFER` is therefore a real change, not a
  formality.
* **FP8 KV cache is likely worth ~1%, not 25%.** Only 10 of the 40 layers keep a KV cache,
  with 2 KV heads × 256 dims each. At 8k context that is ~170 MB read per token versus ~6 GB
  of active weights, so KV bandwidth is nowhere near the bottleneck.
* **The real lever is MTP**: it cuts the number of *target forwards* per output token.
  At the 22 ms/token premise, 2.5× needs a K=3 acceptance rate around 60%. The harness
  measures acceptance from vLLM's Prometheus counters, so the number falls out of the run.

---

## 1. What I found while reading the checkpoints and vLLM 0.18.0

### 1.1 MTP weights are absent from WeMM

| Checkpoint | keys | `mtp.*` keys | declared `mtp_num_hidden_layers` |
| --- | ---: | ---: | ---: |
| `tencent/WeMM-Video-35BA3B` (BF16) | 32,211 | **0** | 1 |
| `tencent/WeMM-Video-35BA3B-AWQ` | 93,891 | **0** | 1 |
| `Qwen/Qwen3.5-35B-A3B` (base) | 1,811 | **785** | 1 |

The text configs are dimension-identical (`hidden_size 2048`, 40 layers, 256 experts,
8 active, `moe_intermediate_size 512`, vocab 248,320, hybrid 3:1 GDN/full attention). The
MTP head was simply stripped when the Tencent release was packaged.

### 1.2 vLLM 0.18.0 does support Qwen3.5 MTP — and it accepts a separate draft path

* `SpeculativeConfig` lists `qwen3_5_mtp` in `MTPModelTypes` and `Qwen3_5MoeMTP` is
  registered in the model registry (`Qwen3_5MTP` / `Qwen3_5MoeMTP` in `qwen3_5_mtp.py`).
* When a draft `ModelConfig` is built, vLLM passes
  `hf_overrides=SpeculativeConfig.hf_config_override`. That function looks at the *draft*
  config's top-level `model_type`; `qwen3_5_moe` becomes `qwen3_5_mtp` with architecture
  `Qwen3_5MoeMTP` and `n_predict = mtp_num_hidden_layers`.
* Consequence: the sidecar's `config.json` must be Qwen's (top-level `model_type:
  qwen3_5_moe`), not WeMM's (`wemm_qwen3_5_moe`). The target still loads WeMM; only the draft
  reads the sidecar. That is why `--speculative-config` can carry an explicit `model` path.

The sidecar built by `bench/make_mtp_head.py` contains:

```
mtp_head.safetensors          mtp.*  (785 tensors, from Qwen3.5-35B-A3B)
                              lm_head.weight, model.embed_tokens.weight  (from WeMM)
model.safetensors.index.json
config.json                   Qwen3.5-35B-A3B config (model_type qwen3_5_moe)
MANIFEST.json                 provenance + dimension check
```

WeMM's `lm_head`/`embed_tokens` are used deliberately: the MTP predictor consumes the target
hidden state and token embedding, so target-consistent weights maximise acceptance. The
builder refuses to run if the two text configs disagree on any structural dimension.

### 1.3 Does the graft change quality?

No. Speculative decoding never accepts a draft token the target would not have produced
(greedy: target argmax; sampled: rejection sampling). Worst case the grafted head is bad and
the acceptance rate drops toward zero, at which point the run is simply no faster than
non-speculative. The report proves this per arm with a token-identical comparison against the
baseline on four greedy prompts.

### 1.4 CUDA graphs / FlashInfer / FP8 KV

* `enforce_eager` defaults to `False` → CUDA graphs are captured. The `eager` diagnostic arm
  exists to quantify what they are worth on this GPU.
* vLLM 0.18 backend priority (`vllm/platforms/cuda.py`): for compute capability major 10
  (B100/B200) FlashInfer is first; **otherwise `FLASH_ATTN` is first**, then FlashInfer, then
  Triton. The RTX Pro 6000 Blackwell reports major 12, so the auto-selector will try
  FlashAttention 2 first. Upstream FA2 has no sm_120 kernels, so the realistic outcomes are
  "falls through to FlashInfer" or "falls through to Triton" — both slower or less
  predictable than pinning FlashInfer. Hence the explicit arm.
* Linear-attention layers use a separate `MambaAttentionBackendEnum` (`LINEAR`/`GDN_ATTN`), so
  `--attention-backend FLASHINFER` only affects the 10 full-attention layers. Good.
* FP8 KV: `kv_cache_dtype` applies to the full-attention layers only; the GDN recurrent state
  stays FP32 via `--mamba-ssm-cache-dtype float32` (as the official recipe already specifies).
  The arm is included because it was requested, but see the arithmetic in §3.
* The official recipe is `--tensor-parallel-size 2`. A single 96 GB card must use TP=1. The
  baseline uses TP=1 too, so the comparison is still apples-to-apples; only the reduction
  order differs from the published recipe.

---

## 2. Measurement protocol

Definition, exactly as specified:

```
decode_tps = (completion_tokens - 1) / (end_to_end_latency - ttft)
```

* Streaming `POST /v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`,
  `max_tokens=256`, `chat_template_kwargs.enable_thinking=false`.
* TTFT = client-side time from sending the request to the first chunk that carries a
  generated token. E2E = time to the final content/finish chunk. Output tokens come from
  `usage.completion_tokens` (`stream_options.include_usage`).
  The client reads SSE lines through `http.client.readline`, not
  `requests.iter_lines`, because the latter fills a 512-byte buffer and inflated measured
  TTFT by ~700 ms in testing.
* Prompts: two short (~40–60 tokens) and two long (~3,000 tokens, standing in for a
  video-length context), interleaved each repetition, all unique.
* 2 discarded warmups per arm, then 5 repetitions per prompt (20 measured requests),
  0.2 s apart. Median and p10/p90 are reported; the headline is the median over all requests.
* One fresh server per arm, prefix caching disabled for every arm, same `max-model-len`,
  same GPU-memory fraction.
* `nvidia-smi` is sampled at 1 Hz during the measured loop (utilisation, memory, power,
  SM clock, temperature) and summarised per arm.
* Speculative acceptance is read from `/metrics`
  (`vllm:spec_decode_num_accepted_tokens_total / num_draft_tokens_total`), not guessed.
* Quality: four greedy, non-streaming prompts (math, factual, arithmetic, long-context
  recall). Outputs are compared against the baseline as strings **and** as token-id
  sequences obtained from `/tokenize`.

The report also parses the server log to record which attention backend, CUDA-graph state,
speculative config, TP size and KV dtype were actually used — flags are requests, logs are
evidence.

---

## 3. Why 2.5× has to come from MTP

At the premise's 22 ms/token (≈45 tok/s), the device is not bandwidth-bound:

* Active weights are ~3 B parameters → ~6 GB of BF16 reads per token. At the card's
  ~1.8 TB/s that is ~3.3 ms, i.e. ~15% of the 22 ms.
* The 10 full-attention layers read ~170 MB of KV per token at 8k context → ~0.1 ms.
  FP8 KV halves that: a ~0.05 ms/token saving, well under 1%.
* The rest of the 22 ms is per-token overhead: 40 layers of MoE routing, expert gathers,
  GDN kernels, launch overhead. This is why CUDA graphs and FlashInfer help, and why a
  weight-quantisation-only story would disappoint on this model at batch 1.

MTP attacks the dominant term: fewer target forwards. With target step time `T`, draft cost
`d` per drafted token, acceptance `a`, and `K` drafts:

```
per_token_latency = (T + K*d) / (1 + a*K)
```

At `T = 22 ms`, `d = 0.8 ms` (one of 40 layers plus routing):

| K | acceptance | ms/token | speedup |
| ---: | ---: | ---: | ---: |
| 3 | 0.50 | 9.8 | 2.25× |
| 3 | 0.60 | 8.7 | 2.52× |
| 3 | 0.70 | 7.9 | 2.79× |
| 5 | 0.45 | 8.0 | 2.75× |
| 5 | 0.35 | 9.2 | 2.39× |

So **~60% acceptance at K=3 is the threshold**. The K sweep (`--full` runs K=3/5/7) plus the
reported acceptance rate turns this into a measurement instead of a guess.

The unknown is whether a head trained alongside the *base* Qwen model still agrees with the
post-trained WeMM model. `qwen3_5_mtp` is a single MoE decoder layer, so the downside is
bounded: bad acceptance just means a slow arm, never wrong output.

---

## 4. Arms

| Arm | Extra flags | Question it answers | Output claim |
| --- | --- | --- | --- |
| `baseline` | *(none)* | Out-of-the-box BF16, official recipe, TP=1 | reference |
| `flashinfer` | `--attention-backend FLASHINFER` | Is pinning FlashInfer worth anything on sm_120? | exact |
| `flashinfer_fp8kv` | `+ --kv-cache-dtype fp8` | Does FP8 KV help at all here? | likely exact |
| `mtp3` | `flashinfer + --speculative-config {model,mtp,3}` | The main lever, K=3 | exact |
| `mtp3_fp8kv` | `flashinfer + fp8 KV + MTP K=3` | The full listed optimisation set | likely exact |
| `ngram` | `flashinfer + n-gram spec K=4` | Zero-extra-weights speculation | exact |
| `flashinfer_fp8kv_scaled` | `+ --calculate-kv-scales` | FP8 KV with runtime scales | likely exact |
| `mtp5`, `mtp7` | `MTP K=5 / K=7` | Acceptance-vs-depth sweep (`--full`) | exact |
| `single_seq` | `mtp3 + --max-num-seqs 1` | Single-user scheduling (`--full`) | exact |
| `eager` | `--enforce-eager` | What CUDA graphs are worth (`--full`) | reference |
| `custom` | `--custom-flags "…"` | Operator-supplied experiment | not claimed |

`claim_exact` drives the report's verdict column: an `exact` arm that diverges at all is
reported as **UNEXPECTED divergence**; a `likely` arm that diverges is called out as expected
FP8 risk.

---

## 5. Running it on the GPU box

Prerequisites: the patched vLLM 0.18.0 environment from the WeMM repo (`patch/install.sh
vllm 0.18.0`), FFmpeg, HF access, ~150 GB free disk, and the 35B BF16 checkpoint either
cached or downloadable.

```bash
# 0. from this directory, on the GPU box
export BENCH_PYTHON=/path/to/.venv-vllm/bin/python   # if `python` is not that venv
./run_bench.sh env          # GPU / vLLM / FlashInfer / WeMM-patch probe (exit 1 on problems)
./run_bench.sh dry-run      # prints the exact `vllm serve` command for every arm

# 1. build the MTP sidecar (~10 GB download from Qwen, ~3.7 GB on disk)
./run_bench.sh prep-mtp

# 2. measure. ~45-75 min: 6 servers, each loading 67 GiB of BF16 weights
./run_bench.sh run

# 3. the report is written automatically; rebuild any time with
./run_bench.sh report
```

`run` asks for confirmation before touching the GPU (skip with `--yes`). It refuses to start
if another process is using more than 3 GiB of GPU memory, and only ever kills the process
group it started.

Useful variants:

```bash
./run_bench.sh run --full                  # K sweep, n-gram, eager, single-seq, scaled FP8
./run_bench.sh run --quick                 # 2 reps per prompt
./run_bench.sh run --arms baseline,mtp3    # a subset
./run_bench.sh run --gpu-mem 0.95          # if BF16 + MTP sidecar OOMs at 0.90
./run_bench.sh run --custom-flags "--moe-backend ..." --arms custom
./run_bench.sh self-test                   # no GPU: mock server, validates the whole pipeline
```

Output layout:

```
results/
  env.json              environment probe
  run_meta.json         invocation + config
  <arm>.json            raw per-request records, quality outputs, metrics, GPU samples
  logs/<arm>.log        full server stdout/stderr
  report.md             the human-readable report
  summary.json          machine-readable summary
  mtp_head/             the sidecar checkpoint
```

The first `run` downloads/loads 67 GiB; if the checkpoint is not cached, pre-fetch it with
`hf download tencent/WeMM-Video-35BA3B` to keep the per-arm startup timing meaningful.
Startup time is recorded separately and never counted in throughput.

---

## 6. Reading the report

* **Decode TPS (median)** — headline number, median over all measured requests.
* **Speedup** — ratio of medians vs `baseline`.
* **Short ctx / Long ctx** — per-bucket medians; a long-context regression that the pooled
  median hides shows up here.
* **TTFT (ms)** — reported because it is subtracted in the metric; MTP barely changes it.
* **Accept. / Mean accepted length** — from vLLM's Prometheus counters; the second column is
  expected tokens per target forward (`1 + accepted/drafts`).
* **Exact-out** — token-identical rate vs baseline on the four quality prompts.
* **Speedup by prompt** — per-prompt ratios, the most robust estimate.
* **What actually ran** — parsed from the log; if `Attention backend` says `TRITON_ATTN`
  while you asked for `FLASHINFER`, the arm fell back and you should read the warnings.
* **Target of 2.5×: MET / NOT MET** — computed from the best successful arm.

`--full` also runs `eager`, which is the honest way to report what CUDA graphs contribute:
`baseline / eager` is the graph benefit, `flashinfer / baseline` the backend benefit, and so
on.

---

## 7. If the numbers fall short

Ordered by what I would try first, given the analysis above:

1. **Look at the acceptance rate.**
   * `≥60%` at K=3 → the 2.5× arithmetic works; if the measured speedup is lower, the draft
     step is more expensive than modelled and K=2 may win. Run `--full` and compare K=1/3/5/7.
   * `<40%` → the post-training drift between Qwen base and WeMM is too large. Options:
     `ngram` speculation (exact, but weak on free-form prose), or a draft from the WeMM
     family (`WeMM-Video-2B`, same 248,320 vocab) via a plain draft-model spec config — that
     needs a small patch because the draft is a multimodal `WeMMVideoForConditionalGeneration`,
     so it is not wired into the harness yet. Ask me and I will add it.
2. **Check `What actually ran`.** A silent `TRITON_ATTN` fallback or `cuda_graphs: False`
   costs more than any flag here gains. The harness retries `flashinfer` once without the pin
   if the backend is rejected, and records `fallback` in the arm JSON.
3. **Drop FP8 KV if it diverges.** It is the only arm that can move tokens, and its expected
   gain is under 1%. `mtp3` (no FP8) is the arm that is exactly output-preserving.
4. **OOM at 0.90 GPU memory.** BF16 weights (67.4 GiB) + sidecar (~3.7 GiB) leave ~10 GiB for
   activations, graphs and KV at 0.90 of 96 GB. Use `--gpu-mem 0.95`, or drop
   `--max-model-len` to 4096 for the benchmark (the long prompts are ~3k tokens).
5. **Weight quantisation is the fallback lever, with a quality caveat.** AWQ (24.4 GB) or
   NVFP4 would cut active-weight traffic ~4×. It is not bit-exact, so it belongs in a separate
   arm with the quality table reviewed, not in the "no quality impact" claim. The WeMM AWQ
   checkpoint also has no MTP head, so it would need the same graft plus AWQ-quantised MTP
   tensors — more fragile. If the BF16 + MTP path reaches ~2× and you want 2.5×, this is the
   next place to look.

---

## 8. Additional optimisations worth trying (not in the original list)

1. **`--max-num-seqs 1` + `--max-num-batched-tokens 8192`** for single-user latency: fewer
   CUDA graphs to capture, no batching headroom to waste. `single_seq` does the first half.
2. **`--async-scheduling`** — reduces scheduler bubbles between steps; cheap to test with
   `--custom-flags`.
3. **MoE/linear backend selection** (`--moe-backend ...` in 0.18). On Blackwell the CUTLASS /
   FlashInfer MoE kernels can beat the default Triton path; this is a per-model, per-shape
   question, so measure it with `--custom-flags` rather than assume.
4. **Compilation config** (`--compilation-config`, `-O3`-style levels) for the language model.
   The hybrid GDN stack may only partially compile; the harness will report a failure rather
   than silently measuring eager.
5. **Tree speculation / deeper MTP chains** if acceptance stays high at K=5–7 — vLLM supports
   a `speculative_token_tree`, which trades a little verification cost for more accepted
   tokens per forward.
6. **Prefix caching + a fixed system prompt** would cut TTFT in production, but it is disabled
   here because the WeMM recipe disables it and because it does not change decode TPS.
7. **CUDA graph mode tuning** (`--compilation-config` cudagraph settings) once the baseline is
   known; full-graph decode can remove the remaining piecewise boundaries.

---

## 9. Repository layout

```
run_bench.sh                 single entrypoint (env / dry-run / prep-mtp / run / report / self-test)
bench/
  common.py                  paths, JSON, port/GPU helpers, Prometheus parsing
  prompts.py                 deterministic prompt suite + quality prompts
  client.py                  streaming client; TTFT/E2E/decode-TPS measurement
  server.py                  vLLM launch/health/metrics/log-evidence, GPU sampler
  variants.py                RunConfig + arm definitions (edit here to add an arm)
  measure.py                 one arm: warmup → quality → measured loop → teardown
  report.py                  report.md + summary.json (incl. quality diff and speedups)
  run_all.py                 orchestrator (arms, env probe, MTP prep, self-test)
  make_mtp_head.py           builds the MTP sidecar from Qwen + WeMM checkpoints
  env_check.py               environment probe
  mock_server.py             OpenAI-compatible mock used by --self-test
tests/test_mtp_builder.py    unit test for the sidecar builder (synthetic checkpoints)
results/selftest/            simulated demo output (clearly marked, not a measurement)
```

Everything is stdlib + `requests` (already in the vLLM venv); only the sidecar builder needs
`torch`, `safetensors` and `huggingface_hub`, which the serving environment has.

---

## 10. Self-test (no GPU)

`./run_bench.sh self-test` starts a mock OpenAI server per arm, streams tokens at a per-arm
simulated rate, serves `/health`, `/metrics`, `/tokenize`, and runs the *real* client,
measurement loop, metrics scrape and report generator. It proves the plumbing; it says
nothing about the GPU.

The simulated demo in `results/selftest/report.md` (K=3 acceptance fixed at 72% by
construction) looks like this:

```
baseline             45.4 tok/s   1.00x
flashinfer           47.5 tok/s   1.05x   100% exact
flashinfer_fp8kv     48.5 tok/s   1.07x   100% exact
mtp3                 81.7 tok/s   1.80x   100% exact   (72% acceptance)
mtp3_fp8kv           83.1 tok/s   1.83x   100% exact
ngram                49.9 tok/s   1.10x   100% exact
```

Treat only the *shape* of that table as meaningful.

Unit test for the sidecar builder (works without a GPU, needs torch + safetensors):

```bash
python -m unittest tests.test_mtp_builder -v
```

---

## 11. Open questions / what I need back

The real run is the only thing left. Please send back (or paste) `results/report.md`,
`results/summary.json` and, if any arm failed, the tail of `results/logs/<arm>.log`. Useful
extras: exact GPU model and driver, and whether the MTP server printed acceptance at
startup.

The two assumptions I could not execute here are:

1. that the patched vLLM accepts the Qwen MTP sidecar as a draft for WeMM (the code path says
   yes; the run will confirm it), and
2. that FlashInfer works on sm_120 in this build (the harness has a one-shot fallback and
   records what happened).

Once the results are in, fill the section below — or tell me and I will do the analysis.

---

## 12. Results

*Not yet measured.* This section will hold the baseline and optimised decode-TPS numbers,
the acceptance rate, the exact-output verification, and the verdict against the 2.5× target.
Run `./run_bench.sh run` on the GPU box and `results/report.md` will contain the same table.

| Arm | Decode TPS | Speedup | Exact-out | Notes |
| --- | ---: | ---: | ---: | --- |
| `baseline` | — | 1.00× | reference | pending |
| `flashinfer` | — | — | — | pending |
| `flashinfer_fp8kv` | — | — | — | pending |
| `mtp3` | — | — | — | pending |
| `mtp3_fp8kv` | — | — | — | pending |
