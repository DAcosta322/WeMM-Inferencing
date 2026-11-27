"""Backport the vLLM 0.31.0 fix for Qwen3.5-MoE config construction.

vLLM 0.18.0 ships ``vllm/transformers_utils/configs/qwen3_5_moe.py`` with

    kwargs["ignore_keys_at_rope_validation"] = [
        "mrope_section",
        "mrope_interleaved",
    ]

transformers 5.2.0 then does ``list | {"partial_rotary_factor"}`` inside
``convert_rope_params_to_dict`` and raises

    TypeError: unsupported operand type(s) for |: 'list' and 'set'

vLLM 0.31.0 uses a set literal instead. This script applies that one-line
backport idempotently (with a ``.orig`` backup), or reports that it is needed.

Why only the MTP draft hits it: the WeMM target's config is handled by the
Tencent patch's own config class, while the MTP sidecar is a stock Qwen3.5-MoE
``config.json`` and goes through this class.

Usage (with the serving venv python):

    python -m bench.fix_vllm_config --check
    python -m bench.fix_vllm_config --apply
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

BUGGY_RE = re.compile(
    r'(?P<key>kwargs\["ignore_keys_at_rope_validation"\]\s*=\s*)'
    r'\[(?P<body>\s*"mrope_section",\s*"mrope_interleaved",\s*)\]'
)
FIXED_RE = re.compile(
    r'kwargs\["ignore_keys_at_rope_validation"\]\s*=\s*\{[^}]*"mrope_section"'
)


def find_config_file() -> Path | None:
    """Locate vLLM's qwen3_5_moe.py without importing vllm (no CUDA needed)."""
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return (
        Path(list(spec.submodule_search_locations)[0])
        / "transformers_utils"
        / "configs"
        / "qwen3_5_moe.py"
    )


def status(path: Path | None = None) -> str:
    path = path or find_config_file()
    if path is None or not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace")
    if FIXED_RE.search(text):
        return "fixed"
    if BUGGY_RE.search(text):
        return "buggy"
    return "unknown"


def apply_fix(path: Path | None = None, backup: bool = True) -> tuple[bool, str]:
    path = path or find_config_file()
    if path is None or not path.exists():
        return False, "vllm/transformers_utils/configs/qwen3_5_moe.py not found"
    current = status(path)
    if current == "fixed":
        return True, f"already fixed: {path}"
    if current != "buggy":
        return False, f"unrecognised file layout, not touched: {path}"
    text = path.read_text(encoding="utf-8")
    if backup:
        backup_path = path.with_suffix(path.suffix + ".orig")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
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
    group.add_argument("--check", action="store_true", help="report whether the fix is needed")
    group.add_argument("--apply", action="store_true", help="apply the backport (default)")
    group.add_argument("--revert", action="store_true", help="restore the .orig backup")
    parser.add_argument("--path", default=None, help="override the qwen3_5_moe.py path")
    parser.add_argument("--validate", default=None, help="MTP sidecar directory to load afterwards")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else find_config_file()
    if path is None:
        print("error: vllm is not importable from this Python", file=sys.stderr)
        return 2

    if args.check:
        state = status(path)
        print(f"{state}: {path}")
        return 0 if state in ("fixed", "unknown") else 1

    if args.revert:
        ok, message = revert(path)
        print(message)
        return 0 if ok else 1

    ok, message = apply_fix(path)
    print(message)
    if not ok:
        return 1
    print(f"status: {status(path)}")
    if args.validate:
        ok, message = validate(Path(args.validate))
        print(("validate: " if ok else "validate FAILED: ") + message)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
