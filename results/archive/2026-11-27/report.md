# WeMM-Video-35B-A3B decode TPS benchmark

Generated: 2026-11-27T06:05:09+00:00

## Result

- Baseline decode TPS: **153.2 tok/s**
- Best arm: **flashinfer_fp8kv** at **159.8 tok/s**, **1.04×** vs baseline
- Target of 2.5×: **NOT MET**



## Arms

| Arm | Status | Decode TPS (median) | Speedup | Short ctx | Long ctx | TTFT (ms) | Accept. | Exact-out | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | ok | 153.2 | 1.00× | 153.1 | 153.2 | 227 | – | – | reference |
| `flashinfer` | ok | 159.6 | 1.04× | 160.4 | 159.5 | 204 | – | 50% |  |
| `flashinfer_fp8kv` | ok | 159.8 | 1.04× | 159.7 | 159.8 | 198 | – | 0% |  |
| `mtp3` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferenci |
| `mtp3_fp8kv` | failed | – | – | – | – | – | – | – | server exited during startup with code 1.
--- log tail ---
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferenci |
| `ngram` | ok | 125.9 | 0.82× | 125.0 | 156.9 | 180 | 32% | 0% |  |

Decode TPS = `(completion_tokens − 1) / (e2e − ttft)`, median over all measured requests; short/long are the per-context medians.

## Speedup by prompt

| Arm | s1 | l1 | s2 | l2 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flashinfer` | 1.05× | 1.04× | 1.05× | 1.04× | 1.04× |
| `flashinfer_fp8kv` | 1.04× | 1.04× | 1.04× | 1.04× | 1.04× |
| `ngram` | 0.81× | 1.23× | 0.82× | 0.82× | 0.82× |

Ratio of per-prompt median decode TPS to the baseline's median for the same prompt, then the median across prompts. This is the most robust speedup estimate because it cancels prompt-specific effects.

## Speculative decoding

| Arm | Acceptance rate | Mean accepted length | Drafts | Accepted |
| --- | ---: | ---: | ---: | ---: |
| `ngram` | 32.3% | 2.29 | 720 | 930 |

Mean acceptance length ≈ `1 + accepted/drafts` tokens per target forward. A K=3 run at 70% acceptance approaches ~2.0× over the same non-speculative server.

## Output quality vs baseline

Greedy, non-streaming, `seed=0`, thinking disabled. Token-level prefix ratio is computed with the server tokenizer; 100% means the two outputs are token-identical.

| Arm | Claim | Exact matches | Token prefix (median) | Verdict |
| --- | --- | ---: | ---: | --- |
| `flashinfer` | exact | 2/4 | 99.7% | **UNEXPECTED divergence** |
| `flashinfer_fp8kv` | likely | 0/4 | 86.7% | diverges (expected risk) |
| `ngram` | exact | 0/4 | 22.9% | **UNEXPECTED divergence** |

<details><summary>flashinfer / q_math divergence</summary>

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

<details><summary>flashinfer_fp8kv / q_math divergence</summary>

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

<details><summary>ngram / q_math divergence</summary>

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
2.  **Add the hours:
    $$09:45 + 2 = 11$$
    Current time: 11:45 (11:45)
3.  Add the minutes:
   
```
</details>

## Failed arms

### `mtp3` — failed

```
(APIServer pid=163008)                 ^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 679, in get_config
(APIServer pid=163008)     config_dict, config = config_parser.parse(
(APIServer pid=163008)                           ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 196, in parse
(APIServer pid=163008)     config = config_class.from_pretrained(
(APIServer pid=163008)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 552, in from_pretrained
(APIServer pid=163008)     return cls.from_dict(config_dict, **kwargs)
(APIServer pid=163008)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 714, in from_dict
(APIServer pid=163008)     config = cls(**config_dict)
(APIServer pid=163008)              ^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5_moe.py", line 192, in __init__
(APIServer pid=163008)     self.text_config = self.sub_configs["text_config"](**text_config)
(APIServer pid=163008)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5_moe.py", line 121, in __init__
(APIServer pid=163008)     super().__init__(**kwargs)
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 219, in __init__
(APIServer pid=163008)     kwargs = self.convert_rope_params_to_dict(
(APIServer pid=163008)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163008)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/modeling_rope_utils.py", line 651, in convert_rope_params_to_dict
(APIServer pid=163008)     ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {"partial_rotary_factor"}
(APIServer pid=163008)                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
(APIServer pid=163008) TypeError: unsupported operand type(s) for |: 'list' and 'set'
```

### `mtp3_fp8kv` — failed

```
(APIServer pid=163176)                 ^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 679, in get_config
(APIServer pid=163176)     config_dict, config = config_parser.parse(
(APIServer pid=163176)                           ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 196, in parse
(APIServer pid=163176)     config = config_class.from_pretrained(
(APIServer pid=163176)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 552, in from_pretrained
(APIServer pid=163176)     return cls.from_dict(config_dict, **kwargs)
(APIServer pid=163176)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 714, in from_dict
(APIServer pid=163176)     config = cls(**config_dict)
(APIServer pid=163176)              ^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5_moe.py", line 192, in __init__
(APIServer pid=163176)     self.text_config = self.sub_configs["text_config"](**text_config)
(APIServer pid=163176)                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/vllm/transformers_utils/configs/qwen3_5_moe.py", line 121, in __init__
(APIServer pid=163176)     super().__init__(**kwargs)
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/configuration_utils.py", line 219, in __init__
(APIServer pid=163176)     kwargs = self.convert_rope_params_to_dict(
(APIServer pid=163176)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=163176)   File "/users/d2acosta/WeMM-Inferencing/WeMM-Video/.venv-vllm/lib/python3.12/site-packages/transformers/modeling_rope_utils.py", line 651, in convert_rope_params_to_dict
(APIServer pid=163176)     ignore_keys_at_rope_validation = ignore_keys_at_rope_validation | {"partial_rotary_factor"}
(APIServer pid=163176)                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~
(APIServer pid=163176) TypeError: unsupported operand type(s) for |: 'list' and 'set'
```

## What actually ran

Flags are requests; the server log is the ground truth. These values are parsed from each arm's log after the run.

| Arm | Attention backend | CUDA graphs | Speculative | TP | KV dtype | GPU blocks |
| --- | --- | --- | --- | ---: | --- | ---: |
| `baseline` | FLASHINFER | True | – | 1 | auto | – |
| `flashinfer` | FLASHINFER | True | – | 1 | auto | – |
| `flashinfer_fp8kv` | FLASHINFER | True | – | 1 | fp8 | – |
| `mtp3` | ? | None | – | ? | ? | – |
| `mtp3_fp8kv` | ? | None | – | ? | ? | – |
| `ngram` | FLASHINFER | True | ngram K=4 | 1 | auto | – |

## Server warnings worth reading

- `baseline`: (EngineCore pid=156615) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=156615) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `baseline`: (EngineCore pid=156615) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `baseline`: (EngineCore pid=156615) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `flashinfer`: (EngineCore pid=159503) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `flashinfer`: (EngineCore pid=159503) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `flashinfer`: (EngineCore pid=159503) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `flashinfer`: (EngineCore pid=159503) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `flashinfer_fp8kv`: (EngineCore pid=161422) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `flashinfer_fp8kv`: (EngineCore pid=161422) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `flashinfer_fp8kv`: (EngineCore pid=161422) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `flashinfer_fp8kv`: (EngineCore pid=161422) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.
- `ngram`: (APIServer pid=163307) WARNING 11-27 01:01:37 [vllm.py:725] Async scheduling not supported with ngram-based speculative decoding and will be disabled.
- `ngram`: (EngineCore pid=163526) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
- `ngram`: (EngineCore pid=163526) <frozen importlib._bootstrap_external>:1301: FutureWarning: The cuda.nvrtc module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.nvrtc module instead.

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

