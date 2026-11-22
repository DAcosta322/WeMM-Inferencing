"""Build the MTP sidecar checkpoint for WeMM-Video-35B-A3B.

Why this exists
---------------
Both released WeMM checkpoints (BF16 and AWQ) declare ``mtp_num_hidden_layers: 1``
but contain **no** ``mtp.*`` tensors, so vLLM's ``--speculative-config
'{"method":"mtp"}'`` cannot work against them directly.

The upstream ``Qwen/Qwen3.5-35B-A3B`` checkpoint, which shares the exact same
backbone dimensions, *does* ship the 785-tensor MTP head. This script produces a
small standalone draft-model directory:

    mtp_head.safetensors        mtp.* from Qwen3.5-35B-A3B  (bf16)
                                + lm_head + embed_tokens from WeMM (bf16)
    model.safetensors.index.json
    config.json                 Qwen3.5-35B-A3B config (model_type qwen3_5_moe)

vLLM loads it as an ``Qwen3_5MoeMTP`` draft. Speculative decoding verifies every
draft token against the target, so the generated tokens are unchanged; only the
acceptance rate (and therefore the speedup) depends on the grafted head quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import human_bytes, log, utc_now, write_json

WEMM_DEFAULT = "tencent/WeMM-Video-35BA3B"
QWEN_DEFAULT = "Qwen/Qwen3.5-35B-A3B"


def _require(package: str, hint: str):
    try:
        return __import__(package)
    except ImportError:
        raise SystemExit(
            f"missing dependency '{package}'. Install it in the serving venv "
            f"({hint}) and retry."
        )


def _resolve(repo: str, revision: str | None, allow_patterns: list[str],
             local_files_only: bool) -> Path:
    path = Path(repo)
    if path.exists():
        return path
    hub = _require("huggingface_hub", "pip install huggingface_hub")
    return Path(
        hub.snapshot_download(
            repo_id=repo,
            revision=revision,
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
        )
    )


def _index_path(root: Path) -> Path:
    index = root / "model.safetensors.index.json"
    if not index.exists():
        candidates = sorted(root.glob("*.safetensors.index.json"))
        if not candidates:
            raise SystemExit(f"no safetensors index found in {root}")
        index = candidates[0]
    return index


def _weight_map(index_path: Path) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)["weight_map"]


def _read_config(root: Path) -> dict:
    path = root / "config.json"
    if not path.exists():
        raise SystemExit(f"config.json missing in {root}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


_DIM_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "mtp_num_hidden_layers",
)


def _check_dims(wemm_text: dict, qwen_text: dict) -> None:
    mismatches = []
    for key in _DIM_KEYS:
        a, b = wemm_text.get(key), qwen_text.get(key)
        if a is not None and b is not None and a != b:
            mismatches.append(f"{key}: wemm={a} qwen={b}")
    if mismatches:
        raise SystemExit(
            "Qwen MTP head is not dimension-compatible with the WeMM text config:\n  "
            + "\n  ".join(mismatches)
        )


def build(
    wemm_model: str = WEMM_DEFAULT,
    qwen_model: str = QWEN_DEFAULT,
    out: Path | str = "results/mtp_head",
    force: bool = False,
    local_files_only: bool = False,
    wemm_revision: str | None = None,
    qwen_revision: str | None = None,
) -> Path:
    safetensors = _require("safetensors", "pip install safetensors")
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = Path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(f"{out} already exists and is not empty (use --force to overwrite)")

    log(f"resolving Qwen checkpoint: {qwen_model}")
    qwen_root = _resolve(
        qwen_model, qwen_revision, ["config.json", "*.safetensors.index.json"], local_files_only
    )
    qwen_index = _index_path(qwen_root)
    qwen_map = _weight_map(qwen_index)
    mtp_keys = {k: v for k, v in qwen_map.items() if k.startswith("mtp.")}
    if not mtp_keys:
        raise SystemExit(f"no mtp.* tensors found in {qwen_model} ({qwen_index})")
    qwen_shards = sorted(set(mtp_keys.values()))
    log(f"  {len(mtp_keys)} mtp tensors across {len(qwen_shards)} shard(s)")
    qwen_root = _resolve(qwen_model, qwen_revision, ["config.json", "*.safetensors.index.json", *qwen_shards], local_files_only)

    log(f"resolving WeMM checkpoint: {wemm_model}")
    wemm_root = _resolve(
        wemm_model, wemm_revision, ["config.json", "*.safetensors.index.json"], local_files_only
    )
    wemm_map = _weight_map(_index_path(wemm_root))
    head_keys = {}
    for key in ("lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
        if key in wemm_map:
            head_keys[key] = wemm_map[key]
    if "lm_head.weight" not in head_keys:
        raise SystemExit(f"lm_head.weight not found in {wemm_model}")
    if not any("embed_tokens" in key for key in head_keys):
        raise SystemExit(f"embed_tokens not found in {wemm_model}")
    wemm_shards = sorted(set(head_keys.values()))
    log(f"  using {len(head_keys)} target tensors from {len(wemm_shards)} shard(s)")
    wemm_root = _resolve(
        wemm_model, wemm_revision, ["config.json", "*.safetensors.index.json", *wemm_shards], local_files_only
    )

    # -- dimension check -------------------------------------------------
    wemm_cfg = _read_config(wemm_root)
    qwen_cfg = _read_config(qwen_root)
    _check_dims(_text_config(wemm_cfg), _text_config(qwen_cfg))

    # -- load tensors ----------------------------------------------------
    tensors = {}
    weight_map: dict[str, str] = {}
    log("loading mtp.* tensors from Qwen")
    by_file: dict[str, list[str]] = {}
    for key, fname in mtp_keys.items():
        by_file.setdefault(fname, []).append(key)
    for fname, keys in sorted(by_file.items()):
        log(f"  {fname}: {len(keys)} tensors")
        with safe_open(qwen_root / fname, framework="pt", device="cpu") as handle:
            for key in keys:
                tensors[key] = handle.get_tensor(key).contiguous()
                weight_map[key] = "mtp_head.safetensors"

    log("loading lm_head / embed_tokens from WeMM")
    for key, fname in head_keys.items():
        target = key
        if key == "model.embed_tokens.weight":
            # vLLM's MTP loader expects model.embed_tokens.weight.
            target = key
        elif key == "model.language_model.embed_tokens.weight":
            target = "model.embed_tokens.weight"
        with safe_open(wemm_root / fname, framework="pt", device="cpu") as handle:
            tensors[target] = handle.get_tensor(key).contiguous()
            weight_map[target] = "mtp_head.safetensors"

    out.mkdir(parents=True, exist_ok=True)
    payload = out / "mtp_head.safetensors"
    log(f"writing {payload} ({len(tensors)} tensors)")
    save_file(tensors, str(payload), metadata={"format": "pt"})

    with (out / "model.safetensors.index.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": payload.stat().st_size}, "weight_map": weight_map}, fh, indent=2)
        fh.write("\n")

    # config.json must describe a Qwen3.5-MoE model so that vLLM's
    # SpeculativeConfig.hf_config_override maps it to Qwen3_5MoeMTP.
    with (out / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(qwen_cfg, fh, indent=2)
        fh.write("\n")

    manifest = {
        "created_at": utc_now(),
        "kind": "wemm-mtp-sidecar",
        "source_target": {"repo": str(wemm_model), "revision": wemm_revision, "path": str(wemm_root)},
        "source_mtp": {"repo": str(qwen_model), "revision": qwen_revision, "path": str(qwen_root)},
        "tensors": len(tensors),
        "mtp_tensors": len(mtp_keys),
        "size_bytes": payload.stat().st_size,
        "notes": (
            "mtp.* weights come from Qwen3.5-35B-A3B; lm_head/embed_tokens come from the "
            "WeMM checkpoint. Verified by speculative decoding: target tokens are unchanged."
        ),
    }
    write_json(out / "MANIFEST.json", manifest)
    log(f"sidecar ready: {out} ({human_bytes(payload.stat().st_size)})")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wemm-model", default=WEMM_DEFAULT, help="HF id or local path of the WeMM checkpoint")
    parser.add_argument("--qwen-model", default=QWEN_DEFAULT, help="HF id or local path of Qwen3.5-35B-A3B")
    parser.add_argument("--out", default="results/mtp_head", help="output sidecar directory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing sidecar directory")
    parser.add_argument("--local-files-only", action="store_true", help="do not touch the network")
    parser.add_argument("--wemm-revision", default=None)
    parser.add_argument("--qwen-revision", default=None)
    args = parser.parse_args(argv)
    build(
        wemm_model=args.wemm_model,
        qwen_model=args.qwen_model,
        out=Path(args.out),
        force=args.force,
        local_files_only=args.local_files_only,
        wemm_revision=args.wemm_revision,
        qwen_revision=args.qwen_revision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
