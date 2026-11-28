# WeMM-Video-35B-A3B — Decode-TPS Benchmark Harness

A reproducible benchmark and optimisation harness for measuring and improving the **decode
throughput** of [`tencent/WeMM-Video-35BA3B`](https://huggingface.co/tencent/WeMM-Video-35BA3B)
on a single **NVIDIA RTX PRO 6000 Blackwell** (single-user). It runs a server per
configuration ("arm"), measures decode TPS end-to-end with a streaming client, verifies output
quality, and writes a Markdown report plus a machine-readable `summary.json`.

The current build id is `1.3.0 / 2026-11-28-quant` (`./run_bench.sh help` prints it).

- **Everything is stdlib + `requests`** (already in the vLLM venv). Only the MTP sidecar
  builder additionally needs `torch`, `safetensors` and `huggingface_hub`.
- **No GPU needed for** `env`, `dry-run`, `prep-mtp`, `report`, `self-test`, or `list-arms`.
  Only `run` starts servers.

---

## Contents

- [What it measures](#what-it-measures)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Prerequisites & install](#prerequisites--install)
- [CLI reference](#cli-reference)
- [Arms](#arms)
- [Output & reading the report](#output--reading-the-report)
- [Troubleshooting](#troubleshooting)
- [Further optimisation ideas](#further-optimisation-ideas)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Current status](#current-status)

---

## What it measures

The headline metric is defined exactly as:

```
decode_tps = (completion_tokens − 1) / (end_to_end_latency − ttft)
```

- **Streaming** `POST /v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`,
  fixed `max_tokens`, `chat_template_kwargs.enable_thinking=false`.
- **TTFT** = client time to the first generated token; **E2E** = client time to the final
  content/finish chunk. Output tokens come from `usage.completion_tokens`
  (`stream_options.include_usage`).
- A deterministic prompt suite: two **short** (~40–60 tokens) and two **long**
  (~3k tokens, standing in for video-length context), interleaved across repetitions so
  thermal drift spreads evenly.
- Per arm: **2 discarded warmups**, then **5 repetitions × 4 prompts** (20 measured requests),
  0.2 s apart. The report shows the **median** decode TPS plus p10/p90 and short/long medians.
- **Quality gate**: four greedy non-streaming prompts (math, factual, reasoning, long-context)
  compared against the baseline as both strings and token-id sequences (via `/tokenize`).
- **Evidence, not flags**: the harness parses each server log to record the attention backend,
  CUDA-graph state, speculative config, TP size, KV dtype and quantization that were
  *actually* used; `nvidia-smi` is sampled at 1 Hz during the measured loop; speculative
  acceptance is read from `/metrics`, not guessed.

A full definition of the measurement loop lives in [`bench/measure.py`](bench/measure.py) and
the protocol is printed at the bottom of every generated report.

---

## How it works

The harness is organised around **arms**: one arm = one `vllm serve` invocation with a specific
set of flags, measured with identical prompts, seeds, token limits and protocol. Arms are small
and additive so each optimisation's contribution is visible. See [Arms](#arms) for the full
list and [`bench/variants.py`](bench/variants.py) for the definitions.

### Lossless vs lossy optimisations

- **Speculative decoding (MTP, tree, n-gram)** is *lossless for greedy output*: the target
  re-verifies every draft token, so it changes only the acceptance rate, never the tokens.
  These arms carry `claim=exact`.
- **Quantisation (AWQ / GPTQ / NVFP4 / FP8)** is *lossy*: it changes the weights, so it
  changes the tokens. These arms carry `claim=likely` and their exact-out column is
  informative, not a pass/fail.

### MTP (the grafted Qwen head)

Both released WeMM checkpoints (BF16 and AWQ) declare `mtp_num_hidden_layers: 1` but contain
**zero** `mtp.*` tensors. The sidecar builder [`bench/make_mtp_head.py`](bench/make_mtp_head.py)
grafts the 785-tensor MTP head from upstream `Qwen/Qwen3.5-35B-A3B` (same backbone dimensions)
onto WeMM's own `lm_head` + `embed_tokens`, producing a ~3.7 GB draft model that vLLM loads via
`--speculative-config '{"method":"mtp","model":…}'`. Speculative verification keeps the target
distribution, so quality is unchanged by construction; only the acceptance rate (and therefore
the speedup) depends on the graft.

vLLM 0.18.0 needs three small backports before the MTP and NVFP4 arms can start — all applied by
`./run_bench.sh fix-mtp-config`:

1. `transformers_utils/configs/qwen3_5_moe.py` passes a `list` to
   `ignore_keys_at_rope_validation`; transformers 5.2 expects a `set`.
2. `v1/spec_decode/eagle.py` does not recognise WeMM architectures and reads
   `image_token_index`, which WeMM exposes as `image_token_id`.
3. `model_executor/layers/quantization/modelopt.py` matches quantised layers by the exact
   `quant_algo` string `"NVFP4"`, but Tencent's `-NVFP4` checkpoint (ModelOpt 0.45.0) writes
   `"W4A16_NVFP4"` — so its quantised MoE experts are loaded as BF16 and engine init dies with
   a weight shape mismatch. The backport matches `"NVFP4" in quant_algo`.

All are idempotent, keep `.orig` backups, and change no numerics.

### Weight quantisation

The `<quant>*` arms pass `--quantization <method>` against a quantised checkpoint supplied via
`--quant-model` (a Hugging Face repo id or a local path), with the method chosen by
`--quant-method` (default `awq`). Tencent publishes a checkpoint per format under
`huggingface.co/tencent/`:

| method | vLLM flag | published checkpoint |
|---|---|---|
| `awq` (default) | `awq_marlin` | [`tencent/WeMM-Video-35BA3B-AWQ`](https://huggingface.co/tencent/WeMM-Video-35BA3B-AWQ) (AWQ W4A16) |
| `nvfp4` | `modelopt_fp4` | [`tencent/WeMM-Video-35BA3B-NVFP4`](https://huggingface.co/tencent/WeMM-Video-35BA3B-NVFP4) (NVFP4 W4A16) |
| `fp8` | `fp8` | [`tencent/WeMM-Video-35BA3B-FP8`](https://huggingface.co/tencent/WeMM-Video-35BA3B-FP8) (FP8 Block128) |
| `gptq` | `gptq_marlin` | *none published* — quantise it yourself if you want this comparison |

(There is also a [`tencent/WeMM-Video-35BA3B-GGUF`](https://huggingface.co/tencent/WeMM-Video-35BA3B-GGUF)
for CPU/native runtime, but the patched-vLLM serving path covers BF16, FP8, AWQ and NVFP4 — not
GGUF.)

Quantising the ~3 B *active* expert weights cuts per-token weight traffic ~4× (4-bit) or ~2×
(8-bit), which matters because weight reads are roughly half the measured decode budget. For
this hybrid video model, prefer a method that keeps the recurrent GDN/linear-attention layers
stable (`fp8` is the safe choice there) and leaves the vision tower, embeddings and `lm_head`
unquantised — they are not on the decode-critical path. Only `gptq` requires producing a
checkpoint offline.

### Tree speculation

The `*_tree` arms add a `speculative_token_tree` (depth 3, branching 2 → 14 nodes) to the MTP
speculative config, so one target forward verifies a *tree* of candidates instead of a linear
chain. This hedges near-tie decisions, but the extra verification cost can offset the extra
acceptance when the linear chain already accepts most tokens — measure it rather than assume.

---

## Quick start

```bash
# 1. Probe the environment (GPU / vLLM / FlashInfer / WeMM patch). No GPU needed.
./run_bench.sh env

# 2. Print the exact `vllm serve` command for every arm. No GPU needed.
./run_bench.sh dry-run

# 3. Build the MTP sidecar (~10 GB download from Qwen, ~3.7 GB on disk).
./run_bench.sh prep-mtp

# 4. One-time vLLM 0.18.0 backports (rope config + eagle + modelopt NVFP4).
./run_bench.sh fix-mtp-config

# 5. Measure (starts one server per arm; ~20 min default BF16 suite, ~50 min with quant arms).
./run_bench.sh run

# 6. Rebuild the report from existing results at any time.
./run_bench.sh report
```

Validate the whole pipeline without a GPU:

```bash
./run_bench.sh self-test
```

`run` asks for confirmation before touching the GPU (`--yes` to skip). It refuses to start if
another process is using more than 3 GiB of GPU memory, and only ever kills the process group it
started. If the preferred port is busy it logs the owner and advances to the next free port
(`--strict-port` fails instead).

---

## Prerequisites & install

- The **patched vLLM 0.18.0** environment from the WeMM repo (`patch/install.sh vllm 0.18.0`).
- **FFmpeg**, **CUDA NPP** (torchcodec links it — see below), HF access, **~150 GB disk**, and
  the 35B BF16 checkpoint cached or downloadable.

`./run_bench.sh` uses the Python it is invoked with. That must be the Python with the patched
vLLM; it detects an activated venv automatically and otherwise honours `BENCH_PYTHON`.

### Build the serving environment

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git     # Fedora: dnf install -y ffmpeg git

git clone https://github.com/Tencent/WeMM-Video.git
cd WeMM-Video
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
bash patch/install.sh vllm 0.18.0
python -c "import vllm; print(vllm.__version__)"              # expect 0.18.0

# torchcodec (imported by WeMM at model load) links CUDA NPP, which the PyTorch wheel
# does not bundle. Install it, then put its lib dir on the loader path.
python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87
source scripts/npp-env.sh
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'
```

If the OS does not package Python 3.12 (Ubuntu 25.04+/Debian 13 ship 3.13), use a standalone
3.12 from `uv` — it needs no root and produces the real venv `patch/install.sh` expects:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

cd WeMM-Video
uv venv --python 3.12 --seed .venv-vllm    # --seed installs pip; install.sh calls python -m pip
source .venv-vllm/bin/activate
bash patch/install.sh vllm 0.18.0

python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87
source scripts/npp-env.sh
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'
```

A healthy `./run_bench.sh env` looks like: `vllm: 0.18.0`, `flashinfer: 0.6.6`,
`wemm_registered: true`, `mtp_registered: true`, `npp_lib_present: true`,
`torchcodec_ok: true`, `problems: []`.

### Extract at the harness root

Extract/run this directory **at its root** — the directory that contains `results/` — never
inside a subdirectory such as `dist/`. The harness resolves the sidecar as `results/mtp_head`
relative to its own location, so a copy in `dist/` silently has no MTP arms.

### Slurm / `srun`

`srun` does not parse `VAR=value` prefixes and a `~` inside quotes is not expanded. With the
venv activated you do not need `BENCH_PYTHON` at all (the script detects `$VIRTUAL_ENV`):

```bash
srun --gres=gpu:1 --mem=96G --pty ./run_bench.sh env

# explicit interpreter — absolute path, note $HOME not ~
srun --gres=gpu:1 --mem=96G --pty bash -lc \
  'cd ~/WeMM-Inferencing && BENCH_PYTHON="$HOME/.../.venv-vllm/bin/python" ./run_bench.sh env'

# equivalent, without a shell wrapper
srun --gres=gpu:1 --mem=96G --pty env BENCH_PYTHON="$HOME/.../.venv-vllm/bin/python" ./run_bench.sh env
```

`run` starts a server and must stay inside one allocation for the whole suite (~1 h); request
`--time=2:00:00` and run inside `tmux` so a dropped SSH session does not kill it.

---

## CLI reference

`run_bench.sh` subcommands:

| Command | Purpose |
|---|---|
| `env` | probe GPU / vLLM / FlashInfer / WeMM patch (no GPU) |
| `dry-run` | print the exact `vllm serve` command for every arm (no GPU) |
| `prep-mtp` | build the MTP sidecar (downloads the Qwen MTP head) |
| `run` | measure baseline + optimised arms, then write the report |
| `report` | rebuild the report from existing results |
| `self-test` | validate the pipeline with the mock server (no GPU) |
| `fix-mtp-config` | backport the vLLM 0.18.0 config fixes (rope + eagle + modelopt NVFP4) |
| `list-arms` | show arm names (marks missing prerequisites) |

Common options (passed through to `bench.run_all`):

| Option | Meaning |
|---|---|
| `--full` | add the K=1/5/7 sweep, n-gram, eager, single-seq, scaled-FP8 diagnostics |
| `--quick` | 2 repetitions instead of 5 |
| `--arms a,b,c` | run only the named arms |
| `--mtp-head PATH` | use an existing sidecar instead of `results/mtp_head` |
| `--quant-model PATH` | enable the `<quant>*` arms against a quantised checkpoint |
| `--quant-method M` | `awq` (default) \| `nvfp4` \| `gptq` \| `fp8` |
| `--gpu-mem 0.90` | vLLM `--gpu-memory-utilization` (per GPU) |
| `--tp 2` | tensor parallel / GPU count (official WeMM recipe uses 2) |
| `--port 8000` | preferred port; auto-advances if busy (`--strict-port` to fail) |
| `--custom-flags "…"` | append raw server flags as one ad-hoc arm named `custom` |
| `--skip-preflight` | run even if torchcodec / MTP backports fail (servers may still fail) |

Useful invocations:

```bash
./run_bench.sh run --full                                # sweep + diagnostics
./run_bench.sh run --quick                               # faster smoke
./run_bench.sh run --arms baseline,mtp3                  # a subset
./run_bench.sh run --gpu-mem 0.95                        # if BF16 + MTP sidecar OOMs at 0.90
./run_bench.sh run --quant-model tencent/WeMM-Video-35BA3B-AWQ     # quantised arms (default AWQ)
./run_bench.sh run --quant-method nvfp4 --quant-model tencent/WeMM-Video-35BA3B-NVFP4
./run_bench.sh run --quant-method fp8 --quant-model tencent/WeMM-Video-35BA3B-FP8
./run_bench.sh run --custom-flags "--moe-backend ..." --arms custom
./run_bench.sh self-test
```

---

## Arms

Default suite (prerequisite arms are dropped automatically if not provided). `<quant>` is the
selected `--quant-method` (default `awq`):

| Arm | Flags | Kind | Claim |
|---|---|---|---|
| `baseline` | *(none)* | baseline | reference |
| `<quant>` | `--quantization <method>` (needs `--quant-model`) | opt | likely |
| `mtp3` | `--speculative-config {model,mtp,3}` | opt | exact |
| `mtp3_tree` | `+ speculative_token_tree` (depth 3, branch 2 → 14 nodes) | opt | exact |
| `<quant>_mtp3` | quantised + MTP K=3 | opt | likely |
| `<quant>_tree` | quantised + MTP tree | opt | likely |
| `baseline_recheck` | identical to `baseline`, run last | opt | noise floor |

`--full` additionally adds `mtp1`, `mtp5`, `mtp7`, `mtp3_fp8kv`, `single_seq`, `fp8kv`,
`fp8kv_scaled`, `ngram` (negative result), `pin_flashinfer` (diagnostic) and `eager`
(diagnostic).

Methodology notes baked into the defaults:

- **No explicit attention backend.** vLLM auto-selects TRITON with CUDA graphs on this GPU; an
  explicit `--attention-backend FLASHINFER` pin crashed engine init on the rebuilt environment.
  If a custom arm passes it and the engine dies, the harness retries once without it.
- **Quality is judged against a noise floor.** Greedy output is not bit-stable across server
  restarts, so the suite ends with `baseline_recheck` and the report compares both full-output
  prefix and the first 64 tokens against it.

---

## Output & reading the report

A `run` writes under `results/`:

```
results/
  env.json              environment probe
  run_meta.json         invocation + config
  <arm>.json            raw per-request records, quality outputs, metrics, GPU samples
  logs/<arm>.log        full server stdout/stderr
  report.md             the human-readable report
  summary.json          machine-readable summary
  mtp_head/             the MTP sidecar checkpoint
```

A rerun deletes `report.md`, `summary.json` and the `<arm>.json` files it is about to replace,
so a failed run can never leave an older report looking current. Always check the `Generated:`
timestamp before interpreting a report.

Report fields:

- **Decode TPS (median)** — headline number, median over all measured requests.
- **Speedup** — ratio of medians vs `baseline`.
- **Short ctx / Long ctx** — per-bucket medians; a long-context regression shows here.
- **TTFT (ms)** — reported because it is subtracted in the metric.
- **Accept. / Mean accepted length** — from vLLM's Prometheus counters; mean accepted length is
  `1 + accepted/drafts` (expected tokens per target forward). A tree drafts many more tokens per
  step, so its raw acceptance rate reads low while its mean length is the meaningful number.
- **Exact-out** — token-identical rate vs baseline on the four quality prompts, read against the
  noise-floor row (`baseline_recheck`).
- **What actually ran** — parsed from the log (backend, CUDA graphs, speculative config,
  quantization, TP, KV dtype); flags are requests, logs are evidence.

`--full` also runs `eager`, the honest way to report what CUDA graphs contribute:
`baseline / eager` is the graph benefit, `flashinfer / baseline` the backend benefit, etc.

---

## Troubleshooting

### `ARM FAILED: port 8000 is already in use`

```bash
./run_bench.sh env --port 8000     # bench_port_free / bench_port_holder
ss -ltnp 'sport = :8000'          # pid + process name
./run_bench.sh run --port 8010    # immediate workaround
```

The harness advances automatically when the preferred port is busy (`--strict-port` fails
instead). If the holder is a stale harness server (`pgrep -af 'vllm serve'`), kill it — it is
also holding GPU memory.

### `BENCH_PYTHON=...` not found / `execve(): BENCH_PYTHON=...`

Two different mistakes, same symptom:

- Exported a placeholder: `export BENCH_PYTHON=/path/to/.venv-vllm/bin/python`. Fix with
  `unset BENCH_PYTHON`; an activated venv is detected automatically.
- `srun BENCH_PYTHON=... ./run_bench.sh` — srun does not parse `VAR=value`. Use
  `srun ... env BENCH_PYTHON=... ./run_bench.sh`, or drop the variable.

### `report.md` says every arm failed at startup

Check the `Generated:` timestamp first. A stale report means the run did not reach the report
stage (look for `ARM CRASHED` in the terminal output and the per-arm JSON/logs).

### `GPU is busy` on a shared node

The GPU already hosts another process (common on MPS nodes, where Slurm shares a physical GPU).
BF16 WeMM needs about **78,000 MiB free** (69 GiB weights + 3.7 GiB sidecar + KV/graphs); a
37 GiB neighbour leaves ~60 GiB, so it cannot fit. **Do not kill a PID you do not own.** In
order: stop it if it is yours; `--wait-gpu 3600` to poll and start when idle; get another
allocation; or use TP=2 across both GPUs (`--tp 2 --gpu-mem 0.55 --allow-busy-gpu`, ~44 GiB
free per GPU, but measures 2 GPUs not the single-GPU target).

`--allow-busy-gpu` with TP=1 skips the check but vLLM will OOM next to a 37 GiB tenant; it is
only useful for small or quantised checkpoints. `./run_bench.sh env` reports
`gpu_memory_free_mib` and warns below 78,000 MiB.

### `unknown arm(s): …`

You are likely running a **stale copy** or from a subdirectory (e.g. `dist/`), where
`results/mtp_head` does not exist. Extract at the harness root, run
`./run_bench.sh list-arms` (it marks MTP arms with `[needs sidecar: …]` and quant arms with
`[needs --quant-model]`), and check `ls results/mtp_head`.

### Every arm after `baseline` fails with `Engine core initialization failed`

Those arms are passing `--attention-backend FLASHINFER`, which crashed engine init on the
rebuilt environment while the auto-selected TRITON baseline ran. The default arms no longer pin
it; a custom arm that pins it and dies is retried once without it (`fallback` is recorded in the
arm JSON).

### MTP arm fails with `AttributeError: … 'image_token_index'`

vLLM 0.18.0's `eagle.py` does not recognise WeMM architectures. Fix with:

```bash
./run_bench.sh fix-mtp-config
python -m bench.fix_vllm_config --check   # both lines must say fixed
```

### MTP arm fails with `TypeError: unsupported operand type(s) for |: 'list' and 'set'`

Same backport (`fix-mtp-config`): vLLM 0.18.0 passes a `list` to
`ignore_keys_at_rope_validation`; transformers 5.2 wants a `set`. Idempotent, `.orig` backup,
no numeric change. Revert with `python -m bench.fix_vllm_config --revert`.

### `fp8kv` arm fails with `Ninja build failed` in the FlashInfer cache

The FP8 KV path JIT-compiles an FP8 prefill kernel that fails to build on this box; FP8 KV
measured +0.1% anyway. It is a `--full`-only negative result.

### `nvfp4` arm fails with `RuntimeError: start (0) + length (512) exceeds dimension size (256)`

Tencent's `-NVFP4` checkpoint (ModelOpt 0.45.0) writes `quant_algo: "W4A16_NVFP4"`, but
vLLM 0.18.0's `modelopt_mixed` loader only recognises the exact string `"NVFP4"`. The
quantised MoE experts are therefore loaded as plain BF16 and the weight load fails on the
packed 4-bit shape. Fix (idempotent, `.orig` backup, no numeric change):

```bash
./run_bench.sh fix-mtp-config
python -m bench.fix_vllm_config --check   # modelopt W4A16_NVFP4 must say fixed
```

This also makes the harness run `--quantization modelopt_mixed` (not `modelopt_fp4`) for the
`nvfp4` method, matching the checkpoint's `MIXED_PRECISION` quant_algo.

### `OSError: libnppicc.so.12: cannot open shared object file`

WeMM's processor imports `torchcodec.decoders` at model load, and torchcodec's CUDA build links
CUDA NPP, which the PyTorch wheel does not bundle. Install it **and** put its lib dir on the
loader path (a bare `import torchcodec` keeps failing until then — that is expected):

```bash
# inside the serving venv
python -m pip install --index-url https://pypi.org/simple nvidia-npp-cu12==12.4.1.87
source scripts/npp-env.sh
python -c 'from torchcodec.decoders import VideoDecoder, AudioDecoder; print("torchcodec ok")'
```

`./run_bench.sh` exports that path for the whole harness, and `bench.common.runtime_env()`
injects it into every server process and the `env` probe, so inside the harness no manual export
is needed. If CUDA comes from a module instead of a wheel, the equivalent is
`export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"`.

---

## Further optimisation ideas

Not in the default suite; cheap to test with `--custom-flags`:

1. **`--max-num-seqs 1` + `--max-num-batched-tokens 8192`** for single-user latency.
2. **`--async-scheduling`** — reduces scheduler bubbles between steps.
3. **MoE/linear backend selection** (`--moe-backend …`) — CUTLASS/FlashInfer MoE kernels can
   beat the default Triton path on Blackwell; measure rather than assume.
4. **Compilation config** (`--compilation-config`, `-O3`-style levels) — the hybrid GDN stack
   may only partially compile.
5. **Deeper MTP chains / tree speculation** if acceptance stays high at K=5–7.
6. **Prefix caching + a fixed system prompt** cuts TTFT in production (disabled here; it does
   not change decode TPS).
7. **CUDA graph mode tuning** once the baseline is known.
8. **Patched vLLM 0.31.0** — carries the Qwen3.5-MoE config fix natively and newer Blackwell
   kernels, at the cost of a new environment (PyTorch 2.13).

---

## Repository layout

```
run_bench.sh                 single entrypoint (env / dry-run / prep-mtp / run / report / self-test)
scripts/npp-env.sh           source this to put CUDA NPP on LD_LIBRARY_PATH for manual commands
bench/
  common.py                  paths, JSON, port/GPU helpers, Prometheus parsing
  prompts.py                 deterministic prompt suite + quality prompts
  client.py                  streaming client; TTFT/E2E/decode-TPS measurement
  server.py                  vLLM launch/health/metrics/log-evidence, GPU sampler
  fix_vllm_config.py         backports the vLLM 0.18.0 config fixes (rope + eagle + modelopt)
  variants.py                RunConfig + arm definitions (edit here to add an arm)
  measure.py                 one arm: warmup → quality → measured loop → teardown
  report.py                  report.md + summary.json (incl. quality diff and speedups)
  run_all.py                 orchestrator (arms, env probe, MTP prep, self-test)
  make_mtp_head.py           builds the MTP sidecar from Qwen + WeMM checkpoints
  env_check.py               environment probe
  mock_server.py             OpenAI-compatible mock used by --self-test
tests/
  test_mtp_builder.py        sidecar builder (synthetic checkpoints; needs torch/safetensors)
  test_gpu_sampler.py        regression test for thread teardown
  test_variants.py           arm definitions, speculation tree, quantisation flags
  test_port_selection.py     port auto-advance / strict-port behaviour
  test_vllm_fix.py           the three vLLM 0.18.0 backports
results/selftest/            simulated demo output (clearly marked, not a measurement)
refsrc/                      unpacked vLLM 0.18.0 + WeMM sources used to author the patches
```

---

## Testing

```bash
# Unit tests (stdlib + requests; test_mtp_builder additionally needs torch + safetensors)
python -m unittest discover -s tests

# End-to-end pipeline validation with the mock server (no GPU)
./run_bench.sh self-test
```

`self-test` starts a mock OpenAI server per arm, streams tokens at a per-arm simulated rate,
serves `/health`, `/metrics` and `/tokenize`, and runs the *real* client, measurement loop,
metrics scrape and report generator. It proves the plumbing; it says nothing about the GPU —
the resulting `results/selftest/report.md` is marked **SIMULATED**.

---

## Current status

- **Measured baseline: 159.8 tok/s (6.3 ms/token)** on the RTX Pro 6000 Blackwell, TRITON +
  CUDA graphs; a same-config recheck reproduced 159.9 with identical output.
- **MTP has not yet produced a measurement** — vLLM 0.18.0 gaps (now backported via
  `fix-mtp-config`) blocked it, and the acceptance-rate rerun is the remaining unknown.
- **FP8 KV and n-gram speculation are negative results** (no gain / slower + diverging); they
  are kept only as `--full` diagnostics.
- **Quantisation and tree arms are implemented but unmeasured** — the `<quant>*` arms need
  `--quant-model` (+ `--quant-method`) and the MTP/tree arms need the sidecar.

For the per-run numbers, read `results/report.md` (or `results/summary.json`), which the
harness regenerates on every run.
