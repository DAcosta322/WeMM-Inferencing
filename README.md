# WeMM-Video-35B-A3B — single-GPU decode-TPS optimisation harness

`idea.md` is unchanged. Everything I produced lives here, and this file is the write-up:
findings, what the harness does, how to run it on the RTX Pro 6000 Blackwell, and what to do
with the numbers.

**Status:** harness complete, unit-tested, and validated end-to-end against a mock server.
First clean run on the target GPU is in (§12): baseline **153.2 tok/s**, FlashInfer + CUDA
graphs already active, FP8 KV and n-gram rejected on measurement. MTP — the only lever that
can reach 2.5× — was blocked by a vLLM 0.18.0 config bug; the backport is in and one targeted
rerun is outstanding. Numbers marked *simulated* come from the mock server and are not
measurements.

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
* **Measured baseline: 153.2 tok/s (6.5 ms/token)** with FlashInfer and CUDA graphs
  already active. The 2.5× target is 383 tok/s, i.e. 2.6 ms/token.
* **Pinning FlashInfer changed nothing measurable.** The baseline arm was already parsed as
  `FLASHINFER`, so the +4% of the `flashinfer` arm is run-order/clock drift, not the pin.
  A same-config `baseline_recheck` arm now runs last to quantify that noise floor.
* **FP8 KV cache is measurably a dead end**: +0.1% over FlashInfer and 0/4 token-identical
  quality prompts (86.7% prefix). Not worth the quality risk; drop it.
* **n-gram speculation is a dead end on this model**: 0.82× (slower, even at 32% acceptance)
  and 0/4 token-identical outputs. It is kept only as a `--full` negative result.
* **The real lever is MTP**: it cuts the number of *target forwards* per output token.
  At 6.5 ms/token, 2.5× needs a K=3 acceptance rate around 60%. The harness measures
  acceptance from vLLM's Prometheus counters, so the number falls out of the run.
* **MTP was blocked by a vLLM 0.18.0 bug**, not by the graft: its Qwen3.5-MoE config passes
  a list where transformers 5.2 expects a set (`TypeError: ... 'list' and 'set'`). vLLM
  0.31.0 has the fix; `./run_bench.sh fix-mtp-config` backports it in place. Rerun pending.

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

**vLLM 0.18.0 quirk, found on the first real run:** its
`vllm/transformers_utils/configs/qwen3_5_moe.py` does

```python
kwargs["ignore_keys_at_rope_validation"] = ["mrope_section", "mrope_interleaved"]  # list
...
kwargs.setdefault("partial_rotary_factor", 0.25)
```

and transformers 5.2.0 then evaluates `list | {"partial_rotary_factor"}`, raising
`TypeError: unsupported operand type(s) for |: 'list' and 'set'`. vLLM 0.31.0 uses a **set**
literal. The WeMM *target* is unaffected because the Tencent patch routes its config through
its own class; the sidecar uses stock Qwen's config and hits it. `./run_bench.sh
fix-mtp-config` applies the one-line backport (with a `.orig` backup and a config-load
validation), and the run preflights it so MTP arms cannot silently fail again.

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

The clean run measured **153.2 tok/s, i.e. 6.5 ms/token** (baseline arm, 20 requests,
TTFT 227 ms median, FlashInfer + CUDA graphs active). The 2.5× target is therefore
383 tok/s, or 2.6 ms/token. What occupies the current 6.5 ms?

* Active weights are ~3 B parameters → ~6 GB of BF16 reads per token. At 6.5 ms that is
  ~920 GB/s, roughly half of the card's ~1.8 TB/s peak. Weight traffic matters here — the
  22 ms/token premise in the brief was pessimistic — but it is still only part of the budget.
* The remaining time is per-token overhead: 40 layers of MoE routing and expert gathers, GDN
  kernels, launches. This is why CUDA graphs and FlashInfer matter, and why MTP — which
  removes whole target forwards — is the only listed lever that can plausibly reach 2.5×.
* KV traffic is not the bottleneck: the 10 full-attention layers, 8k tokens, 2 KV heads ×
  256 dims, bf16 ≈ 170 MB/token → ~26 GB/s at this speed. FP8 KV halves that, under 1% of
  the total. The measurement agrees: FP8 KV added 0.09% over FlashInfer.

MTP attacks the dominant term: fewer target forwards. With target step time `T`, draft cost
`d` per drafted token, acceptance `a`, and `K` drafts:

```
per_token_latency = (T + K*d) / (1 + a*K)
```

At the measured `T = 6.5 ms` and `d = 0.3 ms` (one of 40 layers plus routing overhead):

| K | acceptance | ms/token | speedup |
| ---: | ---: | ---: | ---: |
| 3 | 0.50 | 2.96 | 2.20× |
| 3 | 0.60 | 2.64 | 2.46× |
| 3 | 0.70 | 2.39 | 2.72× |
| 5 | 0.35 | 2.91 | 2.24× |
| 5 | 0.45 | 2.46 | 2.64× |

So **~60% acceptance at K=3 is the threshold**, and the K sweep (`--full` runs K=3/5/7) plus
the reported acceptance rate turns this into a measurement instead of a guess.

The unknown is whether a head trained alongside the *base* Qwen model still agrees with the
post-trained WeMM model. `qwen3_5_mtp` is a single MoE decoder layer, so the downside is
bounded: bad acceptance just means a slow arm, never wrong output.

## 4. Arms

| Arm | Extra flags | Question it answers | Measured |
| --- | --- | --- | --- |
| `baseline` | *(none)* | Out-of-the-box BF16, official recipe, TP=1 | 153.2 tok/s |
| `flashinfer` | `--attention-backend FLASHINFER` | Is pinning FlashInfer worth anything on sm_120? | 159.6 (+4%, drift) |
| `flashinfer_fp8kv` | `+ --kv-cache-dtype fp8` | Does FP8 KV help at all here? | 159.8, tokens drift |
| `mtp3` | `flashinfer + --speculative-config {model,mtp,3}` | The main lever, K=3 | blocked → backport, rerun |
| `mtp3_fp8kv` | `flashinfer + fp8 KV + MTP K=3` | The full listed optimisation set | blocked → backport, rerun |
| `baseline_recheck` | identical to `baseline`, run last | Noise floor for clocks and greedy output | new, pending |
| `ngram` | `flashinfer + n-gram spec K=4` | Zero-extra-weights speculation | 125.9 (0.82×), diverged |
| `flashinfer_fp8kv_scaled` | `+ --calculate-kv-scales` | FP8 KV with runtime scales | `--full` |
| `mtp5`, `mtp7` | `MTP K=5 / K=7` | Acceptance-vs-depth sweep | `--full` |
| `single_seq` | `mtp3 + --max-num-seqs 1` | Single-user scheduling | `--full` |
| `eager` | `--enforce-eager` | What CUDA graphs are worth | `--full` |
| `custom` | `--custom-flags "…"` | Operator-supplied experiment | not claimed |

The report no longer treats "not bit-identical" as a defect by itself: greedy output on this
model is not bit-stable across server restarts, which the baseline-vs-FlashInfer comparison
demonstrated (same backend, 2/4 identical, 99.7% prefix). Verdicts are now relative to the
`baseline_recheck` noise floor, and a same-config rerun is part of the default suite.

---

## 5. Running it on the GPU box

Prerequisites: the patched vLLM 0.18.0 environment from the WeMM repo (`patch/install.sh
vllm 0.18.0`), FFmpeg, CUDA NPP (torchcodec links against it — see below), HF access,
~150 GB free disk, and the 35B BF16 checkpoint either cached or downloadable.

### 5.1 First-time setup (once per machine)

`./run_bench.sh` uses whatever Python it is invoked with. It must be the Python that has the
patched vLLM. Running the probe under the system interpreter is a valid way to see the GPU,
but it reports `vllm: null` and exits 1 — that is the probe doing its job, not a harness bug.

Look for an existing environment before building one:

```bash
command -v vllm
find / -maxdepth 6 -type f -path '*/bin/vllm' 2>/dev/null | head
docker images 2>/dev/null | grep -i wemm
```

If nothing turns up, build the patched environment. The WeMM installer requires an
**activated Python 3.12 venv on Linux x86_64** and applies the model patch to installed vLLM:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git
# Fedora: sudo dnf install -y ffmpeg git

git clone https://github.com/Tencent/WeMM-Video.git
cd WeMM-Video
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
bash patch/install.sh vllm 0.18.0
python -c "import vllm; print(vllm.__version__)"        # expect 0.18.0

# torchcodec (imported by WeMM's processor at model load) links CUDA NPP, which
# the PyTorch wheel does not bundle. The wheel ships libnppicc.so.12 but dlopen
# does not search site-packages, so the lib dir must be on the loader path.
python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87
source scripts/npp-env.sh          # or let ./run_bench.sh do it for you
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'
```

If the distribution does not package Python 3.12 (Ubuntu 25.04+ and Debian 13 ship 3.13 as
`python3`, and `apt` has no `python3.12`), use a standalone 3.12 from `uv`. It needs no root,
no PPA, and produces a real venv, which is what `patch/install.sh` checks for:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

cd WeMM-Video
uv venv --python 3.12 --seed .venv-vllm    # --seed installs pip; install.sh calls python -m pip
source .venv-vllm/bin/activate
python -c "import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:2] == (3, 12), sys.version"
bash patch/install.sh vllm 0.18.0

# same NPP requirement as above
python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87
source scripts/npp-env.sh
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'
```

Verify before building anything expensive: `python -V` must be 3.12.x and
`python -m pip --version` must work inside the activated venv.

Then, from this benchmark directory (activation is optional — the harness looks for `vllm`
next to `BENCH_PYTHON` — but it keeps `vllm` on PATH and is the least surprising):

```bash
source /path/to/WeMM-Video/.venv-vllm/bin/activate
export BENCH_PYTHON="$(command -v python)"
./run_bench.sh env          # expect vllm 0.18.0, flashinfer 0.6.6, weMM/mtp registered: true
```

A healthy probe looks like: `torch: 2.10.0`, `cuda: 12.9`, `vllm: 0.18.0`,
`flashinfer: 0.6.6`, `wemm_registered: true`, `mtp_registered: true`,
`npp_package: nvidia-npp-cu12`, `npp_lib_present: true`, `torchcodec_ok: true`,
`problems: []`.

Docker alternative: the repo also ships `wemm-video:vllm-0.18.0-cu129`. In that case run the
harness *inside* the container with this directory and `results/` bind-mounted, and set
`BENCH_PYTHON` to the container's Python — the harness only needs to be able to launch
`vllm serve` locally.

### 5.2 Slurm / `srun`

`srun` does not interpret `VAR=value` prefixes: it execs the first non-option token, so
`srun BENCH_PYTHON=... ./run_bench.sh` fails with `execve(): BENCH_PYTHON=...: No such file
or directory`. And a `~` inside quotes is not expanded by bash (`~/venv/bin/python` stays
literal). With the venv activated you do not need `BENCH_PYTHON` at all — the script detects
`$VIRTUAL_ENV`. If you do need it, pass it through a shell, or use `env`:

```bash
# venv already active: the script picks $VIRTUAL_ENV/bin/python automatically
srun --gres=gpu:1 --mem=96G --pty ./run_bench.sh env

# explicit interpreter, absolute path (note $HOME, not ~)
srun --gres=gpu:1 --mem=96G --pty bash -lc \
  'cd ~/WeMM-Inferencing && BENCH_PYTHON="$HOME/WeMM-Inferencing/WeMM-Video/.venv-vllm/bin/python" ./run_bench.sh env'

# equivalent, without a shell wrapper
srun --gres=gpu:1 --mem=96G --pty env BENCH_PYTHON="$HOME/WeMM-Inferencing/WeMM-Video/.venv-vllm/bin/python" ./run_bench.sh env
```

If you see `BENCH_PYTHON resolves to '/path/to/.venv-vllm/bin/python'`, you exported the
README's old placeholder. Run `unset BENCH_PYTHON` (the current script would fall back to the
active venv with a warning, but unsetting is cleaner).

`env`, `dry-run`, `report` and `prep-mtp` do not need the GPU. `run` starts a server and must
stay inside one allocation for the whole suite (~1 h): request `--time=2:00:00`.

```bash
# 0. from this directory, on the GPU box, with the venv active.
#    Do NOT export a placeholder here: an active venv is detected automatically.
#    If you previously exported BENCH_PYTHON=/path/to/..., run `unset BENCH_PYTHON`.
./run_bench.sh env          # GPU / vLLM / FlashInfer / WeMM-patch probe (exit 1 on problems)
./run_bench.sh dry-run      # prints the exact `vllm serve` command for every arm

# 1. build the MTP sidecar (~10 GB download from Qwen, ~3.7 GB on disk)
./run_bench.sh prep-mtp

# 1b. one-time vLLM 0.18.0 backport so the MTP draft config can be constructed
./run_bench.sh fix-mtp-config

# 2. measure. ~45-75 min: 6 servers, each loading 67 GiB of BF16 weights
#    --port is a preference: if it is busy the harness logs the owner and moves
#    to the next free port (--strict-port fails instead)
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

Shared / Slurm nodes: the harness probes and samples only the GPUs in
`CUDA_VISIBLE_DEVICES`, so another user's GPU on the same node will not trip the idle
check. If port 8000 is already bound (common on a shared node), the harness picks the next
free port and the client follows it automatically. For a ~1 hour run, start it inside a
multiplexer so a dropped SSH session does not kill the allocation:

```bash
srun --gres=gpu:1 --mem=128G --pty tmux new -s wemm
# then, inside tmux:
./run_bench.sh env && ./run_bench.sh run
```

96 GB of host RAM is usually enough (weights are mmapped and streamed to the GPU), but
128 GB leaves room for page cache and the MTP sidecar.

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

A rerun deletes `report.md`, `summary.json` and the `<arm>.json` files it is about to
replace, so a failed or interrupted run can never leave an older report looking current.
Always check the `Generated:` timestamp at the top of `report.md` before interpreting it.

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
* **Exact-out** — token-identical rate vs baseline on the four quality prompts. Read it
  next to the **noise floor** line (baseline vs `baseline_recheck`): small drift is normal
  on this model, only larger divergences indicate a real quality change.
* **Speedup by prompt** — per-prompt ratios, the most robust estimate.
* **What actually ran** — parsed from the log; if `Attention backend` says `TRITON_ATTN`
  while you asked for `FLASHINFER`, the arm fell back and you should read the warnings.
* **Target of 2.5×: MET / NOT MET** — computed from the best successful arm.

`--full` also runs `eager`, which is the honest way to report what CUDA graphs contribute:
`baseline / eager` is the graph benefit, `flashinfer / baseline` the backend benefit, and so
on.

---

## 7. What the measurements say to do next

1. **Apply the MTP backport and rerun the MTP arms — they are the only path to 2.5×.**
   ```bash
   ./run_bench.sh fix-mtp-config          # list -> set in vLLM's qwen3_5_moe.py
   ./run_bench.sh run --port 8010 --full --arms baseline,baseline_recheck,mtp3,mtp5,mtp7
   ```
   Look at the acceptance column first: `≥60%` at K=3 makes the arithmetic work; `<40%` means
   the grafted Qwen head did not transfer and weight quantisation (see item 5) or a WeMM-family
   draft model is the next lever.
2. **Stop pinning FlashInfer and calling it a win.** The baseline was already `FLASHINFER`
   with CUDA graphs; the +4% is drift. Use `baseline_recheck` (now in the default suite) as
   the reference, and compare MTP against the mean of baseline/recheck.
3. **Do not use FP8 KV here.** Measured: +0.09% over FlashInfer, 0/4 token-identical,
   86.7% prefix. It is the only arm that can move tokens at all, and it buys nothing.
4. **Do not use n-gram speculation here.** Measured 0.82× (slower despite 32% acceptance)
   and 0/4 token-identical (22.9% prefix). On this hybrid GDN stack it is both slower and
   wrong; it is kept only as a `--full` negative result.
5. **OOM at 0.90 GPU memory.** BF16 weights (67.4 GiB) + sidecar (~3.7 GiB) leave ~10 GiB for
   activations, graphs and KV at 0.90 of 96 GB. Use `--gpu-mem 0.95`, or drop
   `--max-model-len` to 4096 for the benchmark (the long prompts are ~3.6k tokens).
6. **Weight quantisation is the fallback lever, with a quality caveat.** AWQ (24.4 GB) or
   NVFP4 would cut active-weight traffic ~4×; at 6.5 ms/token the weights are ~50% of the
   budget, so this is now a plausible second lever, unlike at the 22 ms premise. It is not
   bit-exact, so it belongs in a separate arm with the quality table reviewed. The WeMM AWQ
   checkpoint also has no MTP head, so it needs the same graft plus AWQ-quantised MTP
   tensors — more fragile.

### Troubleshooting

#### `ARM FAILED: port 8000 is already in use`

Something else on the box is listening on 8000 — a manually started vLLM, a leftover
server from an interrupted run, Jupyter, a proxy. Every arm fails because every arm asks
for the same port.

```bash
./run_bench.sh env --port 8000     # bench_port_free / bench_port_holder
ss -ltnp 'sport = :8000'          # pid + process name
lsof -nP -iTCP:8000 -sTCP:LISTEN  # alternative if ss is unavailable
./run_bench.sh run --port 8010    # immediate workaround
```

The current harness advances automatically: when the preferred port is busy it logs who
owns it and uses the next free one (8001, 8002, …). `--strict-port` disables that and
fails with the holder's name instead. If the holder is a stale harness server
(`pgrep -af 'vllm serve'`), kill it before re-running — it is also holding GPU memory,
which the next `run` would report as "GPU is busy" with the offending PIDs.

#### `interpreter 'BENCH_PYTHON=...' not found` / `execve(): BENCH_PYTHON=...`

Two different mistakes, same symptom:

* `BENCH_PYTHON` exported to the literal placeholder from older docs:
  `export BENCH_PYTHON=/path/to/.venv-vllm/bin/python`. Fix with `unset BENCH_PYTHON`;
  an activated venv is detected automatically. (The script now falls back with a warning
  instead of aborting, but the variable should still be removed.)
* `srun BENCH_PYTHON=... ./run_bench.sh` — srun does not parse `VAR=value` before the
  command; it tries to exec `BENCH_PYTHON=...` as a program. Use
  `srun ... env BENCH_PYTHON=... ./run_bench.sh`, or drop the variable entirely.

#### `report.md` says every arm failed at startup

Check the `Generated:` timestamp first. If it predates your latest run, you are reading a
stale report: the harness deletes `report.md`/`summary.json` at the start of a run, so a
missing report means the run did not reach the report stage (look for `ARM CRASHED` in the
terminal output and the per-arm JSON/logs). Re-running is what produces a fresh one.

#### MTP arm fails with `TypeError: unsupported operand type(s) for |: 'list' and 'set'`

vLLM 0.18.0's `vllm/transformers_utils/configs/qwen3_5_moe.py` passes a list to
`ignore_keys_at_rope_validation`; transformers 5.2.0 expects a set. It only bites the MTP
draft (stock Qwen config), not the WeMM target. vLLM 0.31.0 already uses a set. Apply the
one-line backport, then rerun:

```bash
./run_bench.sh fix-mtp-config     # writes a .orig backup, patches, validates the sidecar
./run_bench.sh run --port 8010    # preflight now passes; MTP arms start normally
# if you ever need to undo it:
python -m bench.fix_vllm_config --revert
```

The patch is idempotent and doesn't change any numerics — only the container type of a config
validation hint.

#### `OSError: libnppicc.so.12: cannot open shared object file`

This is the failure you hit on the first real run. WeMM's processor imports
`torchcodec.decoders` while the model loads, and torchcodec's CUDA build links against CUDA
NPP, which the PyTorch wheel does not bundle. The log shows two `FFmpeg version 5/4` blocks,
then `Could not load this library: .../libtorchcodec_core5.so`, then every arm fails at
startup (the harness records the tail in `results/<arm>.json` and in the report's
"Failed arms" section).

Installing the wheel is necessary but **not sufficient**: it lands in
`site-packages/nvidia/npp/lib`, and `dlopen` does not search `site-packages`. A bare
`python -c 'import torchcodec'` will keep failing until that directory is on
`LD_LIBRARY_PATH`. This is expected and does not mean the install failed.

```bash
# inside the serving venv
python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87

# if the uv-created venv has no pip:
uv pip install --python "$(command -v python)" \
  --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87

# manual check outside the harness: put the wheel's lib dir on the loader path
source scripts/npp-env.sh
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'

# outside the harness, the equivalent is (both lines, same shell):
#   npp_lib=$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))')
#   export LD_LIBRARY_PATH="$npp_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

./run_bench.sh env --port 8010     # npp_lib_present + torchcodec_ok should now be true
./run_bench.sh run --port 8010
```

Nothing in that snippet is a placeholder: `LD_LIBRARY_PATH` is the variable being assigned,
`$npp_lib` comes from the line above it, and `${LD_LIBRARY_PATH:+...}` only appends the old
value when one already exists. Paste both lines into the same shell, then check:

```bash
echo "$LD_LIBRARY_PATH"    # .../site-packages/nvidia/npp/lib
ls "$LD_LIBRARY_PATH"      # libnppicc.so.12 (and friends)
```

If you have no other CUDA paths to preserve, the whole thing is also just:

```bash
export LD_LIBRARY_PATH="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))')"
```

`./run_bench.sh` exports that path for the whole harness, and `bench/common.runtime_env()`
also injects it into every server process and into the `env` probe — so inside the harness no
manual export is needed. An `export` lasts only for that shell: do it inside the `srun`/`tmux`
session where you start the run, or put `source scripts/npp-env.sh` in the job script. If CUDA comes from a module rather than a wheel, the equivalent is
`export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"`.

The harness now checks this before launching anything and refuses to start six doomed
servers (`--skip-preflight` overrides). If `torchcodec` itself is missing, the probe says so
and points at the pinned `torchcodec==0.10.0+cu129` from the WeMM lock instead.

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
8. **Try patched vLLM 0.31.0** (the WeMM patch supports it for BF16). It carries the
   Qwen3.5-MoE config fix natively, is newer on the Blackwell kernels, and would remove the
   one-line backport from the setup. It is a new environment (PyTorch 2.13) and not needed
   for the MTP rerun.

---

## 9. Repository layout

```
run_bench.sh                 single entrypoint (env / dry-run / prep-mtp / run / report / self-test)
scripts/npp-env.sh           source this to put CUDA NPP on LD_LIBRARY_PATH for manual commands
bench/
  common.py                  paths, JSON, port/GPU helpers, Prometheus parsing
  prompts.py                 deterministic prompt suite + quality prompts
  client.py                  streaming client; TTFT/E2E/decode-TPS measurement
  server.py                  vLLM launch/health/metrics/log-evidence, GPU sampler
  fix_vllm_config.py         backports the vLLM 0.18.0 Qwen3.5-MoE config fix
  variants.py                RunConfig + arm definitions (edit here to add an arm)
  measure.py                 one arm: warmup → quality → measured loop → teardown
  report.py                  report.md + summary.json (incl. quality diff and speedups)
  run_all.py                 orchestrator (arms, env probe, MTP prep, self-test)
  make_mtp_head.py           builds the MTP sidecar from Qwen + WeMM checkpoints
  env_check.py               environment probe
  mock_server.py             OpenAI-compatible mock used by --self-test
tests/test_mtp_builder.py    unit test for the sidecar builder (synthetic checkpoints)
tests/test_gpu_sampler.py    regression test for thread teardown (Thread._stop shadowing)
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

Done and confirmed on the target GPU: FlashInfer is active on sm_120 (both the auto-selected
baseline and the pinned arm report `backend=FLASHINFER`), CUDA graphs are captured, and the
BF16 checkpoint loads and serves at 153 tok/s.

Still open: whether the grafted Qwen MTP head gives usable acceptance on post-trained WeMM.
That needs the backport and one targeted rerun:

```bash
./run_bench.sh fix-mtp-config
./run_bench.sh run --port 8010 --full --arms baseline,baseline_recheck,mtp3,mtp5,mtp7
```

Send `results/report.md` + `results/summary.json` afterwards (or just the arms table). If an
MTP arm fails, `results/logs/mtp3.log` is the file to send.

---

## 12. Results

First clean run on **NVIDIA RTX PRO 6000 Blackwell Server Edition** (97,887 MiB, driver
615.71.10, compute 12.0), vLLM 0.18.0, torch 2.10.0+cu129:

| Arm | Decode TPS | Speedup | Exact-out | Notes |
| --- | ---: | ---: | ---: | --- |
| `baseline` | **153.2** | 1.00× | reference | FLASHINFER + CUDA graphs already active |
| `flashinfer` | 159.6 | 1.04× | 2/4 (99.7% prefix) | same backend as baseline → drift, not the pin |
| `flashinfer_fp8kv` | 159.8 | 1.04× | 0/4 (86.7% prefix) | +0.09% over FlashInfer; tokens moved |
| `mtp3` | — | — | — | failed: vLLM config bug, backport ready, rerun pending |
| `mtp3_fp8kv` | — | — | — | same |
| `ngram` | 125.9 | **0.82×** | 0/4 (22.9% prefix) | slower at 32% acceptance; outputs diverged |

**Target of 2.5× (383 tok/s): not met, and not yet testable** — the only lever that can reach
it (MTP) was blocked by the vLLM config bug. What this run does establish:

1. **The baseline is much faster than the brief assumed.** 6.5 ms/token, not 22 ms, with
   FlashInfer and CUDA graphs already on. The brief's CUDA-graph/FlashInfer items are
   therefore already satisfied by stock vLLM on this GPU.
2. **FP8 KV is not worth it here.** +0.09% over FlashInfer and a visible token drift
   (86.7% prefix, 0/4 identical). It was the one requested optimisation that can change
   output, and it buys nothing.
3. **n-gram speculation is not a fallback on this model.** 0.82× despite 32% acceptance and
   0/4 identical outputs; the hybrid GDN state handling in vLLM's n-gram path looks
   unreliable here.
4. **Greedy output is not bit-stable across server restarts.** Baseline vs pinned FlashInfer
   (identical backend, identical flags in effect) differed on 2/4 prompts at 99.7% prefix.
   That is why the report now uses a `baseline_recheck` noise floor instead of treating any
   difference as a regression.
5. **The MTP blocker was a real vLLM 0.18.0 bug**, reproducible from the traceback alone
   (list vs set in `qwen3_5_moe.py`), fixed in 0.31.0 and backported by
   `./run_bench.sh fix-mtp-config`.

The next run (baseline + recheck + MTP K=3/5/7) is the one that decides the 2.5× claim. At
60% acceptance the arithmetic says ~2.5×; at 40% it says ~2.0×.
