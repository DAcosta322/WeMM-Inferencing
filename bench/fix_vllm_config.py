"""Backports for vLLM 0.18.0 so the WeMM MTP + quantised checkpoints can start.

Three independent defects, all hit on the target GPU:

1. ``vllm/transformers_utils/configs/qwen3_5_moe.py`` passes a **list** to
   ``ignore_keys_at_rope_validation``; transformers 5.2 expects a set and raises
   ``TypeError: unsupported operand type(s) for |: 'list' and 'set'``.
   vLLM 0.31.0 uses a set.

2. ``vllm/v1/spec_decode/eagle.py`` classifies multimodal targets by architecture
   name. ``WeMMVideoMoeForConditionalGeneration`` is not in the Qwen3.5 list, so
   it falls through to ``target_model.config.image_token_index``. WeMM's config
   has ``image_token_id``, so it raises
   ``AttributeError: 'WeMMQwen3_5MoeConfig' object has no attribute 'image_token_index'``.

3. ``vllm/model_executor/layers/quantization/modelopt.py``'s
   ``ModelOptMixedPrecisionConfig`` routes quantised layers by exact ``quant_algo``
   match (``== "NVFP4"``). Tencent's ``-NVFP4`` checkpoint was produced by ModelOpt
   0.45.0 and writes the full name ``W4A16_NVFP4``, so the quantised MoE experts are
   loaded as plain BF16 and engine init dies with
   ``RuntimeError: start (0) + length (512) exceeds dimension size (256)``.

All edits are idempotent, keep a ``.orig`` backup, and do not change numerics.

Usage (with the serving venv python):

    python -m bench.fix_vllm_config --check
    python -m bench.fix_vllm_config --apply --validate results/mtp_head
    python -m bench.fix_vllm_config --revert
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

# --- fix 1: list -> set in qwen3_5_moe.py ------------------------------------
BUGGY_RE = re.compile(
    r'(?P<key>kwargs\["ignore_keys_at_rope_validation"\]\s*=\s*)'
    r'\[(?P<body>\s*"mrope_section",\s*"mrope_interleaved",\s*)\]'
)
FIXED_RE = re.compile(
    r'kwargs\["ignore_keys_at_rope_validation"\]\s*=\s*\{[^}]*"mrope_section"'
)

# --- fix 2: recognise WeMM architectures in eagle.py -------------------------
EAGLE_OLD = (
    '                "Qwen3_5MoeForConditionalGeneration",\n'
    "            ]:"
)
EAGLE_NEW = (
    '                "Qwen3_5MoeForConditionalGeneration",\n'
    '                "WeMMVideoForConditionalGeneration",\n'
    '                "WeMMVideoMoeForConditionalGeneration",\n'
    "            ]:"
)
EAGLE_FIXED_MARKER = "WeMMVideoMoeForConditionalGeneration"

# --- fix 3: recognise ModelOpt 0.45 ``W4A16_NVFP4`` quant_algo -----------------
# get_quant_method routes by exact string match; Tencent's checkpoint says
# "W4A16_NVFP4" (upper-cased). Match by substring so both "NVFP4" and
# "W4A16_NVFP4" resolve to the NVFP4 method.
MODELOPT_NVFP4_OLD = 'quant_algo == "NVFP4"'
MODELOPT_NVFP4_NEW = '"NVFP4" in quant_algo'
MODELOPT_GROUP_OLD = 'layer_info.get("quant_algo", "").upper() == "NVFP4"'
MODELOPT_GROUP_NEW = '"NVFP4" in layer_info.get("quant_algo", "").upper()'
MODELOPT_FIXED_MARKER = '"NVFP4" in quant_algo'


def _vllm_root() -> Path | None:
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def find_config_file() -> Path | None:
    root = _vllm_root()
    return root / "transformers_utils" / "configs" / "qwen3_5_moe.py" if root else None


def find_eagle_file() -> Path | None:
    root = _vllm_root()
    return root / "v1" / "spec_decode" / "eagle.py" if root else None


def find_modelopt_file() -> Path | None:
    root = _vllm_root()
    return (
        root / "model_executor" / "layers" / "quantization" / "modelopt.py"
        if root
        else None
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)


# ---------------------------------------------------------------------------
# fix 1
# ---------------------------------------------------------------------------


def status(path: Path | None = None) -> str:
    path = path or find_config_file()
    if path is None or not path.exists():
        return "missing"
    text = _read(path)
    if FIXED_RE.search(text):
        return "fixed"
    if BUGGY_RE.search(text):
        return "buggy"
    return "unknown"


def apply_fix(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_config_file()
    if path is None or not path.exists():
        return False, "vllm/transformers_utils/configs/qwen3_5_moe.py not found"
    current = status(path)
    if current == "fixed":
        return True, f"already fixed: {path}"
    if current != "buggy":
        return False, f"unrecognised file layout, not touched: {path}"
    text = path.read_text(encoding="utf-8")
    _backup(path)
    new_text, count = BUGGY_RE.subn(lambda m: f"{m.group('key')}{{{m.group('body')}}}", text)
    if count != 1:
        return False, f"expected exactly one match, found {count}; not touched"
    path.write_text(new_text, encoding="utf-8")
    return True, f"patched {path}"


def revert(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_config_file()
    if path is None:
        return False, "not found"
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        return False, f"no backup at {backup}"
    shutil.copy2(backup, path)
    return True, f"reverted {path} from {backup}"


# ---------------------------------------------------------------------------
# fix 2
# ---------------------------------------------------------------------------


def eagle_status(path: Path | None = None) -> str:
    path = path or find_eagle_file()
    if path is None or not path.exists():
        return "missing"
    text = _read(path)
    if EAGLE_FIXED_MARKER in text:
        return "fixed"
    if EAGLE_OLD in text:
        return "buggy"
    return "unknown"


def apply_eagle_fix(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_eagle_file()
    if path is None or not path.exists():
        return False, "vllm/v1/spec_decode/eagle.py not found"
    current = eagle_status(path)
    if current == "fixed":
        return True, f"already fixed: {path}"
    if current != "buggy":
        return False, f"unrecognised file layout, not touched: {path}"
    text = path.read_text(encoding="utf-8")
    _backup(path)
    if text.count(EAGLE_OLD) != 1:
        return False, (
            f"expected one Qwen3.5 architecture list, found {text.count(EAGLE_OLD)}; "
            "not touched"
        )
    path.write_text(text.replace(EAGLE_OLD, EAGLE_NEW), encoding="utf-8")
    return True, f"patched {path}"


def revert_eagle(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_eagle_file()
    if path is None:
        return False, "not found"
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        return False, f"no backup at {backup}"
    shutil.copy2(backup, path)
    return True, f"reverted {path} from {backup}"


# ---------------------------------------------------------------------------
# fix 3
# ---------------------------------------------------------------------------


def modelopt_status(path: Path | None = None) -> str:
    path = path or find_modelopt_file()
    if path is None or not path.exists():
        return "missing"
    text = _read(path)
    if MODELOPT_FIXED_MARKER in text:
        return "fixed"
    if MODELOPT_NVFP4_OLD in text or MODELOPT_GROUP_OLD in text:
        return "buggy"
    return "unknown"


def apply_modelopt_fix(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_modelopt_file()
    if path is None or not path.exists():
        return False, "vllm/model_executor/layers/quantization/modelopt.py not found"
    current = modelopt_status(path)
    if current == "fixed":
        return True, f"already fixed: {path}"
    if current != "buggy":
        return False, f"unrecognised file layout, not touched: {path}"
    text = _read(path)
    n_nvfp4 = text.count(MODELOPT_NVFP4_OLD)
    n_group = text.count(MODELOPT_GROUP_OLD)
    if n_nvfp4 < 2:
        return False, (
            f"expected two 'quant_algo == \"NVFP4\"' sites in get_quant_method, "
            f"found {n_nvfp4}; not touched"
        )
    _backup(path)
    text = text.replace(MODELOPT_NVFP4_OLD, MODELOPT_NVFP4_NEW)
    text = text.replace(MODELOPT_GROUP_OLD, MODELOPT_GROUP_NEW)
    path.write_text(text, encoding="utf-8")
    return True, f"patched {path} ({n_nvfp4} get_quant_method + {n_group} group_size sites)"


def revert_modelopt(path: Path | None = None) -> tuple[bool, str]:
    path = path or find_modelopt_file()
    if path is None:
        return False, "not found"
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        return False, f"no backup at {backup}"
    shutil.copy2(backup, path)
    return True, f"reverted {path} from {backup}"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(sidecar: Path) -> tuple[bool, str]:
    """Build the draft config the way vLLM will, to prove the fix works."""
    try:
        from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig

        config = Qwen3_5MoeConfig.from_pretrained(str(sidecar))
    except Exception as exc:  # noqa: BLE001 - report the real error
        return False, f"config load failed: {type(exc).__name__}: {exc}"
    text = getattr(config, "text_config", None)
    rope = getattr(text, "rope_parameters", None)
    return True, (
        f"loaded {type(config).__name__} / {type(text).__name__}; "
        f"mtp_num_hidden_layers={getattr(text, 'mtp_num_hidden_layers', '?')}; "
        f"rope_parameters={rope}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report whether the fixes are needed")
    group.add_argument("--apply", action="store_true", help="apply both backports (default)")
    group.add_argument("--revert", action="store_true", help="restore the .orig backups")
    parser.add_argument("--path", default=None, help="override qwen3_5_moe.py (testing)")
    parser.add_argument("--eagle-path", default=None, help="override eagle.py (testing)")
    parser.add_argument("--modelopt-path", default=None, help="override modelopt.py (testing)")
    parser.add_argument("--validate", default=None, help="MTP sidecar directory to load afterwards")
    args = parser.parse_args(argv)

    qwen = Path(args.path) if args.path else find_config_file()
    eagle = Path(args.eagle_path) if args.eagle_path else find_eagle_file()
    modelopt = Path(args.modelopt_path) if args.modelopt_path else find_modelopt_file()
    if qwen is None:
        print("error: vllm is not importable from this Python", file=sys.stderr)
        return 2

    if args.check:
        q_state, e_state = status(qwen), eagle_status(eagle)
        m_state = modelopt_status(modelopt)
        print(f"qwen3_5_moe rope params: {q_state}: {qwen}")
        print(f"eagle WeMM image token:  {e_state}: {eagle}")
        print(f"modelopt W4A16_NVFP4:    {m_state}: {modelopt}")
        return 0 if "buggy" not in (q_state, e_state, m_state) else 1

    if args.revert:
        ok1, msg1 = revert(qwen)
        print(msg1)
        ok2, msg2 = revert_eagle(eagle) if eagle else (False, "eagle.py not found")
        print(msg2)
        ok3, msg3 = revert_modelopt(modelopt) if modelopt else (False, "modelopt.py not found")
        print(msg3)
        return 0 if (ok1 and ok2 and ok3) else 1

    ok1, msg1 = apply_fix(qwen)
    print(msg1)
    ok2, msg2 = apply_eagle_fix(eagle)
    print(msg2)
    ok3, msg3 = apply_modelopt_fix(modelopt)
    print(msg3)
    print(
        f"status: qwen={status(qwen)} eagle={eagle_status(eagle)} "
        f"modelopt={modelopt_status(modelopt)}"
    )
    if not (ok1 and ok2 and ok3):
        return 1
    if args.validate:
        ok, message = validate(Path(args.validate))
        print(("validate: " if ok else "validate FAILED: ") + message)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
