"""Unit tests for arm definitions, the speculation tree, and quantisation flags."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from bench.variants import (
    QUANT_METHODS,
    TREE_BRANCHING,
    TREE_DEPTH,
    Arm,
    RunConfig,
    arm_quantization,
    arm_spec_config,
    build_arms,
    quant_flags,
    select_arms,
    tree_node_count,
    tree_spec_string,
)


class TreeSpecTest(unittest.TestCase):
    def test_tree_is_valid_literal_and_sorted_breadth_first(self) -> None:
        s = tree_spec_string(TREE_DEPTH, TREE_BRANCHING)
        nodes = ast.literal_eval(s)
        self.assertEqual(len(nodes), tree_node_count(TREE_DEPTH, TREE_BRANCHING))
        self.assertEqual(len(nodes), 14)  # 2 + 4 + 8
        self.assertEqual(nodes[0], (0,))
        self.assertEqual(nodes[1], (1,))
        self.assertEqual(sorted(nodes, key=lambda t: (len(t), t)), nodes)
        for node in nodes:
            self.assertIsInstance(node, tuple)
            self.assertTrue(all(0 <= i < TREE_BRANCHING for i in node))

    def test_tree_node_count(self) -> None:
        self.assertEqual(tree_node_count(3, 2), 14)
        self.assertEqual(tree_node_count(1, 2), 2)


class QuantMethodTest(unittest.TestCase):
    def test_registry_contains_all_methods(self) -> None:
        self.assertEqual(
            set(QUANT_METHODS), {"awq", "gptq", "nvfp4", "fp8"}
        )
        # AWQ is the recommended default and must come first.
        self.assertEqual(next(iter(QUANT_METHODS)), "awq")

    def test_quant_flags_map_to_vllm_values(self) -> None:
        self.assertEqual(quant_flags("awq"), ["--quantization", "awq_marlin"])
        self.assertEqual(quant_flags("gptq"), ["--quantization", "gptq_marlin"])
        self.assertEqual(quant_flags("nvfp4"), ["--quantization", "modelopt_mixed"])
        self.assertEqual(quant_flags("fp8"), ["--quantization", "fp8"])


class ArmBuildTest(unittest.TestCase):
    def _cfg(self, method: str = "awq") -> RunConfig:
        return RunConfig(
            mtp_head=Path("/tmp/mtp_head"),
            quant_model="/tmp/wemm-quant",
            quant_method=method,
        )

    def test_default_arms_use_awq(self) -> None:
        arms = build_arms(self._cfg(), full=False)
        names = {a.name for a in arms}
        for expected in (
            "baseline", "awq", "mtp3", "mtp3_tree", "awq_mtp3", "awq_tree", "baseline_recheck",
        ):
            self.assertIn(expected, names)

    def test_gptq_method_produces_gptq_arm_names(self) -> None:
        arms = build_arms(self._cfg("gptq"), full=False)
        names = {a.name for a in arms}
        self.assertIn("gptq", names)
        self.assertIn("gptq_mtp3", names)
        self.assertIn("gptq_tree", names)
        self.assertNotIn("awq", names)

    def test_quant_arm_flags_and_model(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "awq")
        self.assertEqual(arm_quantization(arm), "awq_marlin")
        self.assertEqual(arm.model, "/tmp/wemm-quant")
        self.assertTrue(arm.requires_quant)
        self.assertFalse(arm.requires_mtp)

    def test_tree_arm_spec_has_token_tree(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "mtp3_tree")
        spec = arm_spec_config(arm)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["method"], "mtp")
        self.assertEqual(spec["num_speculative_tokens"], TREE_DEPTH)
        self.assertEqual(len(ast.literal_eval(spec["speculative_token_tree"])), 14)
        self.assertTrue(arm.requires_mtp)
        self.assertFalse(arm.requires_quant)

    def test_quant_mtp3_requires_both(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "awq_mtp3")
        self.assertTrue(arm.requires_mtp)
        self.assertTrue(arm.requires_quant)
        self.assertEqual(arm.model, "/tmp/wemm-quant")
        self.assertEqual(arm_quantization(arm), "awq_marlin")
        spec = arm_spec_config(arm)
        self.assertEqual(spec["num_speculative_tokens"], 3)
        self.assertNotIn("speculative_token_tree", spec)

    def test_arms_dropped_when_prerequisites_missing(self) -> None:
        # quant arms are always built (need --quant-model to run); MTP arms need a
        # sidecar, so they are absent here.
        arms = build_arms(RunConfig(), full=False)
        names = {a.name for a in arms}
        self.assertIn("awq", names)  # requires_quant, dropped by the orchestrator
        self.assertNotIn("mtp3", names)
        self.assertNotIn("awq_mtp3", names)
        self.assertIn("baseline", names)
        self.assertIn("baseline_recheck", names)

    def test_unknown_quant_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_arms(self._cfg("bogus"), full=False)

    def test_select_arms_handles_quant_names(self) -> None:
        arms = build_arms(self._cfg(), full=False)
        picked = select_arms(arms, "awq,mtp3_tree,awq_tree")
        self.assertEqual([a.name for a in picked], ["awq", "mtp3_tree", "awq_tree"])

    def test_arm_to_dict_exposes_spec_and_quantization(self) -> None:
        arm = Arm(
            name="x",
            label="x",
            kind="opt",
            extra_flags=["--quantization", "fp8", "--speculative-config", '{"method":"mtp"}'],
        )
        data = arm.to_dict()
        self.assertEqual(data["quantization"], "fp8")
        self.assertEqual(data["speculative"], {"method": "mtp"})


if __name__ == "__main__":
    unittest.main()
