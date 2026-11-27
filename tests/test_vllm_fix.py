"""Unit test for the vLLM 0.18.0 qwen3_5_moe.py list->set backport."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench.fix_vllm_config import apply_fix, revert, status

BUGGY = '''class Qwen3_5MoeTextConfig:
    def __init__(self, **kwargs):
        kwargs["ignore_keys_at_rope_validation"] = [
            "mrope_section",
            "mrope_interleaved",
        ]
        kwargs.setdefault("partial_rotary_factor", 0.25)
'''


class VllmConfigFixTest(unittest.TestCase):
    def test_apply_is_idempotent_and_revertible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_5_moe.py"
            path.write_text(BUGGY)
            self.assertEqual(status(path), "buggy")

            ok, _ = apply_fix(path)
            self.assertTrue(ok)
            self.assertEqual(status(path), "fixed")
            text = path.read_text()
            self.assertIn('kwargs["ignore_keys_at_rope_validation"] = {', text)
            self.assertTrue((Path(tmp) / "qwen3_5_moe.py.orig").exists())

            ok, _ = apply_fix(path)  # second apply must not fail
            self.assertTrue(ok)

            ok, _ = revert(path)
            self.assertTrue(ok)
            self.assertEqual(path.read_text(), BUGGY)

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
