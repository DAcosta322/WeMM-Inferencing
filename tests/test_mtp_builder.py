"""Unit test for the MTP sidecar builder using tiny synthetic checkpoints.

Run with the serving venv (needs torch + safetensors):

    python -m unittest tests.test_mtp_builder -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from bench.make_mtp_head import build


def _text_config(hidden: int = 8) -> dict:
    return {
        "hidden_size": hidden,
        "num_hidden_layers": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "mtp_num_hidden_layers": 1,
        "tie_word_embeddings": False,
        "model_type": "qwen3_5_moe_text",
    }


def _write_checkpoint(root: Path, model_type: str, tensors: dict, config: dict, shard: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(root / shard), metadata={"format": "pt"})
    weight_map = {name: shard for name in tensors}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2)
    )
    (root / "config.json").write_text(json.dumps(config, indent=2))


class MtpBuilderTest(unittest.TestCase):
    def test_builds_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qwen_cfg = {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": _text_config(),
            }
            wemm_cfg = {
                "architectures": ["WeMMVideoForConditionalGeneration"],
                "model_type": "wemm_qwen3_5_moe",
                "text_config": _text_config(),
            }
            qwen_tensors = {
                "mtp.fc.weight": torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8),
                "mtp.norm.weight": torch.ones(8),
                "mtp.pre_fc_norm_hidden.weight": torch.ones(8),
                "mtp.pre_fc_norm_embedding.weight": torch.ones(8),
                "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
                "mtp.layers.0.self_attn.k_proj.weight": torch.zeros(2, 8),
                "mtp.layers.0.self_attn.v_proj.weight": torch.zeros(2, 8),
                "mtp.layers.0.self_attn.o_proj.weight": torch.zeros(8, 8),
                "mtp.layers.0.input_layernorm.weight": torch.ones(8),
                "mtp.layers.0.post_attention_layernorm.weight": torch.ones(8),
                "mtp.layers.0.mlp.gate.weight": torch.zeros(2, 8),
                "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.zeros(8, 4),
                "mtp.layers.0.mlp.shared_expert_gate.weight": torch.zeros(1, 8),
                "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(8, 4),
                "mtp.layers.0.mlp.experts.1.gate_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.experts.1.up_proj.weight": torch.zeros(4, 8),
                "mtp.layers.0.mlp.experts.1.down_proj.weight": torch.zeros(8, 4),
            }
            wemm_tensors = {
                "lm_head.weight": torch.full((16, 8), 3.0),
                "model.language_model.embed_tokens.weight": torch.full((16, 8), 7.0),
            }
            qwen_root = tmp / "qwen"
            wemm_root = tmp / "wemm"
            _write_checkpoint(qwen_root, "qwen3_5_moe", qwen_tensors, qwen_cfg, "qwen.safetensors")
            _write_checkpoint(wemm_root, "wemm_qwen3_5_moe", wemm_tensors, wemm_cfg, "wemm.safetensors")

            out = tmp / "sidecar"
            build(wemm_model=str(wemm_root), qwen_model=str(qwen_root), out=out, local_files_only=True)

            index = json.loads((out / "model.safetensors.index.json").read_text())
            names = set(index["weight_map"])
            self.assertIn("mtp.fc.weight", names)
            self.assertIn("lm_head.weight", names)
            self.assertIn("model.embed_tokens.weight", names)
            self.assertNotIn("model.language_model.embed_tokens.weight", names)

            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["model_type"], "qwen3_5_moe")

            from safetensors import safe_open

            with safe_open(out / "mtp_head.safetensors", framework="pt", device="cpu") as handle:
                self.assertTrue(torch.equal(handle.get_tensor("lm_head.weight"), wemm_tensors["lm_head.weight"]))
                self.assertTrue(
                    torch.equal(
                        handle.get_tensor("model.embed_tokens.weight"),
                        wemm_tensors["model.language_model.embed_tokens.weight"],
                    )
                )
                self.assertEqual(handle.get_tensor("mtp.fc.weight").shape, (16, 8))
            manifest = json.loads((out / "MANIFEST.json").read_text())
            self.assertEqual(manifest["mtp_tensors"], len(qwen_tensors))

    def test_dimension_mismatch_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qwen_cfg = {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": _text_config(hidden=8),
            }
            wemm_cfg = {
                "architectures": ["WeMMVideoForConditionalGeneration"],
                "model_type": "wemm_qwen3_5_moe",
                "text_config": _text_config(hidden=16),
            }
            _write_checkpoint(
                tmp / "qwen",
                "qwen3_5_moe",
                {"mtp.norm.weight": torch.ones(8)},
                qwen_cfg,
                "qwen.safetensors",
            )
            _write_checkpoint(
                tmp / "wemm",
                "wemm_qwen3_5_moe",
                {
                    "lm_head.weight": torch.zeros(16, 16),
                    "model.language_model.embed_tokens.weight": torch.zeros(16, 16),
                },
                wemm_cfg,
                "wemm.safetensors",
            )
            with self.assertRaises(SystemExit):
                build(
                    wemm_model=str(tmp / "wemm"),
                    qwen_model=str(tmp / "qwen"),
                    out=tmp / "sidecar",
                    local_files_only=True,
                )


if __name__ == "__main__":
    unittest.main()
