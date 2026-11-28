# WeMM-Video-35B-A3B decode TPS benchmark

Generated: 2026-11-28T03:37:35+00:00

## Result

- Baseline decode TPS: **159.8 tok/s**
- Best arm: **baseline_recheck** at **159.9 tok/s**, **1.00×** vs baseline
- Target of 2.5×: **NOT MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 159.8 | 1.00× | 159.0 | 159.8 | 202 | – | – | reference |
| `baseline_recheck` | ok | 159.9 | 1.00× | 160.5 | 159.9 | 202 | – | 100% |  |
| `fp8kv` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:109 |
| `mtp3` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=51760) ERROR 11-27 22:33:29 [core.py:109 |
| `mtp3_fp8kv` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=52102) ERROR 11-27 22:34:37 [core.py:109 |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_recheck` | 1.01× | 1.00× | 1.01× | 1.00× | 1.00× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. `First 64 tok` counts prompts whose first 64 generated token ids match exactly (stable even when a late near-tie flips the tail); `Prefix` is the median token-level common prefix over the full output. Read both against the noise-floor row.

Run-to-run noise floor (identical flags, two servers): **100.0%** token prefix. Greedy output is not bit-stable across server restarts on this model, so small differences are not a quality signal.

| Arm | Claim | Exact | First 64 tok | Prefix (median) | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_recheck` | baseline | 4/4 | 4/4 | 100.0% | noise floor |

## Failed arms

### `fp8kv` — failed

```
server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099]   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/subprocess.py", line 571, in run
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099]     raise CalledProcessError(retcode, process.args,
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099] subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False', '-f', '/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/build.ninja']' returned non-zero exit status 1.
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099]
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099] The above exception was the direct cause of the following exception:
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099]   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 344, in run_ninja
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099]     raise RuntimeError(msg) from e
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099] RuntimeError: Ninja build failed. Ninja output:
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099] ninja: Entering directory `/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False'
(EngineCore pid=51274) ERROR 11-27 22:32:06 [core.py:1099] [1/11] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -isystem /users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/include/python3.12 -isystem /usr/local/cuda/include -isystem /usr/local/cuda/include/cccl -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/tvm_ffi/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/tvm_ffi/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_120a,code=sm_120a -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /users/d2acosta/.cache/flashinfer/0.6.6/120a/generated/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cu -o /users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cuda.o
(EngineCore pid=51274)   File "/users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/subprocess.py", line 571, in run
(EngineCore pid=51274)     raise CalledProcessError(retcode, process.args,
(EngineCore pid=51274) subprocess.CalledProcessError: Command '['ninja', '-v', '-C', '/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False', '-f', '/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/build.ninja']' returned non-zero exit status 1.
(EngineCore pid=51274)
(EngineCore pid=51274) The above exception was the direct cause of the following exception:
(EngineCore pid=51274)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/jit/cpp_ext.py", line 344, in run_ninja
(EngineCore pid=51274)     raise RuntimeError(msg) from e
(EngineCore pid=51274) RuntimeError: Ninja build failed. Ninja output:
(EngineCore pid=51274) ninja: Entering directory `/users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False'
(EngineCore pid=51274) [1/11] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output /users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -isystem /users/d2acosta/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/include/python3.12 -isystem /usr/local/cuda/include -isystem /usr/local/cuda/include/cccl -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/tvm_ffi/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/tvm_ffi/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/csrc -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include -isystem /users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -static-global-template-stub=false -gencode=arch=compute_120a,code=sm_120a -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c /users/d2acosta/.cache/flashinfer/0.6.6/120a/generated/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cu -o /users/d2acosta/.cache/flashinfer/0.6.6/120a/cached_ops/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill_ragged_kernel_mask_3.cuda.o
(APIServer pid=51159)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=51159)     raise RuntimeError(
(APIServer pid=51159) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### `mtp3` — failed

```
server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=51760) ERROR 11-27 22:33:29 [core.py:1099]     return super().__getattribute__(key)
(EngineCore pid=51760) ERROR 11-27 22:33:29 [core.py:1099]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=51760) ERROR 11-27 22:33:29 [core.py:1099] AttributeError: 'WeMMQwen3_5MoeConfig' object has no attribute 'image_token_index'. Did you mean: 'image_token_id'?
(EngineCore pid=51760) Process EngineCore:
(EngineCore pid=51760) Traceback (most recent call last):
(EngineCore pid=51760)     return super().__getattribute__(key)
(EngineCore pid=51760)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=51760) AttributeError: 'WeMMQwen3_5MoeConfig' object has no attribute 'image_token_index'. Did you mean: 'image_token_id'?
[rank0]:[W1127 22:33:30.992932694 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(APIServer pid=51655) Traceback (most recent call last):
(APIServer pid=51655)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=51655)     raise RuntimeError(
(APIServer pid=51655) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

### `mtp3_fp8kv` — failed

```
server exited during startup with code 1.
--- error excerpt ---
(EngineCore pid=52102) ERROR 11-27 22:34:37 [core.py:1099]     return super().__getattribute__(key)
(EngineCore pid=52102) ERROR 11-27 22:34:37 [core.py:1099]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=52102) ERROR 11-27 22:34:37 [core.py:1099] AttributeError: 'WeMMQwen3_5MoeConfig' object has no attribute 'image_token_index'. Did you mean: 'image_token_id'?
(EngineCore pid=52102) Process EngineCore:
(EngineCore pid=52102) Traceback (most recent call last):
(EngineCore pid=52102)     return super().__getattribute__(key)
(EngineCore pid=52102)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=52102) AttributeError: 'WeMMQwen3_5MoeConfig' object has no attribute 'image_token_index'. Did you mean: 'image_token_id'?
[rank0]:[W1127 22:34:38.016782499 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(APIServer pid=51988) Traceback (most recent call last):
(APIServer pid=51988)   File "/users/d2acosta/Projects/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1031, in wait_for_engine_startup
(APIServer pid=51988)     raise RuntimeError(
(APIServer pid=51988) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backends (all logged) | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | TRITON | True | – | 1 | auto | – |
| `baseline_recheck` | TRITON | True | – | 1 | auto | – |
| `fp8kv` | ? | None | – | ? | ? | – |
| `mtp3` | ? | None | – | ? | ? | – |
| `mtp3_fp8kv` | ? | None | – | ? | ? | – |

## Server warnings worth reading

- `baseline`: (EngineCore pid=49327) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=49327) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline`: (EngineCore pid=49327) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=49327) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline_recheck`: (EngineCore pid=52450) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline_recheck`: (EngineCore pid=52450) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline_recheck`: (EngineCore pid=52450) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline_recheck`: (EngineCore pid=52450) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.

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

