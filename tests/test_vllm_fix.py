"""Unit tests for the two vLLM 0.18.0 backports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench.fix_vllm_config import (
    apply_eagle_fix,
    apply_fix,
    eagle_status,
    revert,
    revert_eagle,
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


if __name__ == "__main__":
    unittest.main()
