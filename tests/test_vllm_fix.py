"""Unit tests for the three vLLM 0.18.0 backports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench.fix_vllm_config import (
    apply_eagle_fix,
    apply_fix,
    apply_modelopt_fix,
    eagle_status,
    modelopt_status,
    revert,
    revert_eagle,
    revert_modelopt,
    status,
)

BUGGY = '''class Qwen3_5MoeTextConfig:
    def __init__(self, **kwargs):
        kwargs["ignore_keys_at_rope_validation"] = [
            "mrope_section",
            "mrope_interleaved",
        ]
        kwargs.setdefault("partial_rotary_factor", 0.25)
'''

EAGLE = '''        if supports_multimodal(target_model):
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            else:
                self.model.config.image_token_index = target_model.config.image_token_index
'''


class VllmConfigFixTest(unittest.TestCase):
    def test_rope_fix_is_idempotent_and_revertible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_5_moe.py"
            path.write_text(BUGGY)
            self.assertEqual(status(path), "buggy")
            ok, _ = apply_fix(path)
            self.assertTrue(ok)
            self.assertEqual(status(path), "fixed")
            self.assertIn('kwargs["ignore_keys_at_rope_validation"] = {', path.read_text())
            self.assertTrue((Path(tmp) / "qwen3_5_moe.py.orig").exists())
            self.assertTrue(apply_fix(path)[0])  # second apply must not fail
            self.assertTrue(revert(path)[0])
            self.assertEqual(path.read_text(), BUGGY)

    def test_eagle_fix_adds_wemm_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eagle.py"
            path.write_text(EAGLE)
            self.assertEqual(eagle_status(path), "buggy")
            ok, _ = apply_eagle_fix(path)
            self.assertTrue(ok)
            self.assertEqual(eagle_status(path), "fixed")
            text = path.read_text()
            self.assertIn('"WeMMVideoForConditionalGeneration",', text)
            self.assertIn('"WeMMVideoMoeForConditionalGeneration",', text)
            self.assertTrue(apply_eagle_fix(path)[0])
            self.assertTrue(revert_eagle(path)[0])
            self.assertEqual(path.read_text(), EAGLE)

    def test_unknown_layout_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_5_moe.py"
            path.write_text("# something else\n")
            self.assertEqual(status(path), "unknown")
            ok, message = apply_fix(path)
            self.assertFalse(ok)
            self.assertIn("not touched", message)


MODELOPT = '''class ModelOptMixedPrecisionConfig:
    def _from_config(self):
        for layer_info in quantized_layers.values():
            if layer_info.get("quant_algo", "").upper() == "NVFP4":
                group_size = layer_info.get("group_size", 16)
                break

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            return UnquantizedLinearMethod()

        if isinstance(layer, FusedMoE):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(self.nvfp4_config)
            return None
'''


class ModeloptFixTest(unittest.TestCase):
    def test_w4a16_nvfp4_is_recognised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modelopt.py"
            path.write_text(MODELOPT)
            self.assertEqual(modelopt_status(path), "buggy")
            ok, message = apply_modelopt_fix(path)
            self.assertTrue(ok, message)
            self.assertEqual(modelopt_status(path), "fixed")
            text = path.read_text()
            # get_quant_method: two == "NVFP4" sites become None-safe substring matches.
            self.assertEqual(text.count('quant_algo == "NVFP4"'), 0)
            self.assertIn('quant_algo is not None and "NVFP4" in quant_algo', text)
            self.assertIn('"NVFP4" in layer_info.get("quant_algo", "").upper()', text)
            self.assertTrue(apply_modelopt_fix(path)[0])  # idempotent
            self.assertTrue(revert_modelopt(path)[0])
            self.assertEqual(path.read_text(), MODELOPT)

    def test_partial_backport_is_repaired(self) -> None:
        # A previous non-None-safe backport (`"NVFP4" in quant_algo`) must be
        # repaired to the None-safe form on re-apply.
        intermediate = MODELOPT.replace(
            'quant_algo == "NVFP4"', '"NVFP4" in quant_algo'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modelopt.py"
            path.write_text(intermediate)
            self.assertEqual(modelopt_status(path), "buggy")
            ok, message = apply_modelopt_fix(path)
            self.assertTrue(ok, message)
            text = path.read_text()
            self.assertEqual(text.count('quant_algo == "NVFP4"'), 0)
            self.assertEqual(
                text.count('quant_algo is not None and "NVFP4" in quant_algo'), 2
            )
            self.assertEqual(modelopt_status(path), "fixed")

    def test_unknown_modelopt_layout_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modelopt.py"
            path.write_text("# no match here\n")
            self.assertEqual(modelopt_status(path), "unknown")
            ok, message = apply_modelopt_fix(path)
            self.assertFalse(ok)
            self.assertIn("not touched", message)


if __name__ == "__main__":
    unittest.main()
