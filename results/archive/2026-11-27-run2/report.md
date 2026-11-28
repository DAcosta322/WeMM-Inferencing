# WeMM-Video-35B-A3B decode TPS benchmark

Generated: 2026-11-27T08:52:02+00:00

## Result

- Baseline decode TPS: **153.4 tok/s**
- Best arm: **baseline** at **153.4 tok/s**, **1.00×** vs baseline
- Target of 2.5×: **NOT MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 153.4 | 1.00× | 153.6 | 153.3 | 207 | – | – | reference |
| `baseline_recheck` | ok | 153.4 | 1.00× | 153.6 | 153.3 | 203 | – | 0% |  |
| `flashinfer` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=252332)     args.dispatch_function(args)
(APIS |
| `flashinfer_fp8kv` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=253364)     args.dispatch_function(args)
(APIS |
| `mtp3` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=254284)     args.dispatch_function(args)
(APIS |
| `mtp3_fp8kv` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=255090)     args.dispatch_function(args)
(APIS |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_recheck` | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. Token-level prefix ratio is computed with the server tokenizer; 100% means the two outputs are token-identical.

Run-to-run noise floor (identical flags, two servers): **64.0%** token prefix. Greedy output is not bit-stable across server restarts on this model, so small differences are not a quality signal.

| Arm | Claim | Exact matches | Token prefix (median) | Verdict |
| --- | --- | ---: | ---: | --- |
| `baseline_recheck` | baseline | 0/4 | 64.0% | noise floor |

<details><summary>baseline_recheck / q_math divergence</summary>

```
baseline: To find the end time of the meeting, we add the duration to the start time.

**Arithmetic:**

1.  **Start Time:** 09:45
2.  **Duration:** 2 hours and 35 minutes

**Step 1: Add the hours**
$$09:45 + 2 \text{ hours} = 11:4
arm:      To find the end time of the meeting, we add the duration to the start time.

**Arithmetic:**

1.  **Start Time:** 09:45
2.  **Duration:** 2 hours and 35 minutes

**Step 1: Add the hours**
$$09:45 + 2 \text{ hours} = 11:4
```
</details>

## Failed arms

### `flashinfer` — failed

```
(APIServer pid=252332)            ^^^^
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=252332)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=252332)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=252332)     return func(*args, **kwargs)
(APIServer pid=252332)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 128, in make_async_mp_client
(APIServer pid=252332)     return AsyncMPClient(*client_args)
(APIServer pid=252332)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=252332)     return func(*args, **kwargs)
(APIServer pid=252332)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 924, in __init__
(APIServer pid=252332)     super().__init__(
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 583, in __init__
(APIServer pid=252332)     with launch_core_engines(
(APIServer pid=252332)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=252332)   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=252332)     next(self.gen)
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 972, in launch_core_engines
(APIServer pid=252332)     wait_for_engine_startup(
(APIServer pid=252332)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=252332)     raise RuntimeError(
(APIServer pid=252332) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### `flashinfer_fp8kv` — failed

```
(APIServer pid=253364)            ^^^^
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=253364)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=253364)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=253364)     return func(*args, **kwargs)
(APIServer pid=253364)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 128, in make_async_mp_client
(APIServer pid=253364)     return AsyncMPClient(*client_args)
(APIServer pid=253364)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=253364)     return func(*args, **kwargs)
(APIServer pid=253364)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 924, in __init__
(APIServer pid=253364)     super().__init__(
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 583, in __init__
(APIServer pid=253364)     with launch_core_engines(
(APIServer pid=253364)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=253364)   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=253364)     next(self.gen)
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 972, in launch_core_engines
(APIServer pid=253364)     wait_for_engine_startup(
(APIServer pid=253364)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=253364)     raise RuntimeError(
(APIServer pid=253364) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### `mtp3` — failed

```
(APIServer pid=254284)            ^^^^
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=254284)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=254284)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=254284)     return func(*args, **kwargs)
(APIServer pid=254284)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 128, in make_async_mp_client
(APIServer pid=254284)     return AsyncMPClient(*client_args)
(APIServer pid=254284)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=254284)     return func(*args, **kwargs)
(APIServer pid=254284)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 924, in __init__
(APIServer pid=254284)     super().__init__(
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 583, in __init__
(APIServer pid=254284)     with launch_core_engines(
(APIServer pid=254284)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=254284)   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=254284)     next(self.gen)
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 972, in launch_core_engines
(APIServer pid=254284)     wait_for_engine_startup(
(APIServer pid=254284)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=254284)     raise RuntimeError(
(APIServer pid=254284) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### `mtp3_fp8kv` — failed

```
(APIServer pid=255090)            ^^^^
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py", line 154, in __init__
(APIServer pid=255090)     self.engine_core = EngineCoreClient.make_async_mp_client(
(APIServer pid=255090)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=255090)     return func(*args, **kwargs)
(APIServer pid=255090)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 128, in make_async_mp_client
(APIServer pid=255090)     return AsyncMPClient(*client_args)
(APIServer pid=255090)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
(APIServer pid=255090)     return func(*args, **kwargs)
(APIServer pid=255090)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 924, in __init__
(APIServer pid=255090)     super().__init__(
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 583, in __init__
(APIServer pid=255090)     with launch_core_engines(
(APIServer pid=255090)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=255090)   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=255090)     next(self.gen)
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 972, in launch_core_engines
(APIServer pid=255090)     wait_for_engine_startup(
(APIServer pid=255090)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=255090)     raise RuntimeError(
(APIServer pid=255090) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backend | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | 1 | auto | – |
| `baseline_recheck` | FLASHINFER | True | – | 1 | auto | – |
| `flashinfer` | ? | None | – | ? | ? | – |
| `flashinfer_fp8kv` | ? | None | – | ? | ? | – |
| `mtp3` | ? | None | – | ? | ? | – |
| `mtp3_fp8kv` | ? | None | – | ? | ? | – |

## Server warnings worth reading

- `baseline`: (EngineCore pid=250442) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=250442) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline`: (EngineCore pid=250442) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=250442) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline_recheck`: (EngineCore pid=256014) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline_recheck`: (EngineCore pid=256014) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline_recheck`: (EngineCore pid=256014) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline_recheck`: (EngineCore pid=256014) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.

## Environment

- `gpu_name`: NVIDIA RTX PRO 6000 Blackwell Server Edition
- `gpu_memory_total_mib`: 97887
- `driver`: 615.71.10
- `compute_cap`: 12.0
- `torch`: 2.10.0+cu129
- `cuda`: 12.9.79
- `vllm`: 0.18.0
- `flashinfer`: 0.6.6
- `python`: 3.12.16

## Protocol

- Streaming `/v1/chat/completions`, `temperature=0`, `seed=0`, `ignore_eos=true`, `max_tokens` fixed per arm, `chat_template_kwargs.enable_thinking=false`.
- TTFT = client time to first generated token; E2E = client time to the final content/finish chunk; output tokens from `usage.completion_tokens`.
- 2 discarded warmups per arm, 5 measured repetitions per prompt, prompts interleaved by repetition to spread thermal drift.
- One server per arm, started fresh and stopped afterwards; prefix caching disabled for every arm.

