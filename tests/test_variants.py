"""Unit tests for arm definitions, the speculation tree, and GPTQ flags."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from bench.variants import (
    TREE_BRANCHING,
    TREE_DEPTH,
    Arm,
    RunConfig,
    arm_quantization,
    arm_spec_config,
    build_arms,
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
        # Every node is a tuple path, root's children come first (breadth-first).
        self.assertEqual(nodes[0], (0,))
        self.assertEqual(nodes[1], (1,))
        self.assertEqual(sorted(nodes, key=lambda t: (len(t), t)), nodes)
        # Every child path is one element longer than its parent and uses valid indices.
        for node in nodes:
            self.assertIsInstance(node, tuple)
            self.assertTrue(all(0 <= i < TREE_BRANCHING for i in node))

    def test_tree_node_count(self) -> None:
        self.assertEqual(tree_node_count(3, 2), 14)
        self.assertEqual(tree_node_count(1, 2), 2)


class ArmBuildTest(unittest.TestCase):
    def _cfg(self) -> RunConfig:
        return RunConfig(
            mtp_head=Path("/tmp/mtp_head"),
            gptq_model="/tmp/wemm-gptq",
        )

    def test_default_arms_include_gptq_and_tree(self) -> None:
        arms = build_arms(self._cfg(), full=False)
        names = {a.name for a in arms}
        for expected in ("baseline", "gptq", "mtp3", "mtp3_tree", "gptq_mtp3", "gptq_tree", "baseline_recheck"):
            self.assertIn(expected, names)

    def test_gptq_arm_flags_and_model(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "gptq")
        self.assertIn("--quantization", arm.extra_flags)
        self.assertIn("gptq_marlin", arm.extra_flags)
        self.assertEqual(arm.model, "/tmp/wemm-gptq")
        self.assertTrue(arm.requires_gptq)
        self.assertEqual(arm_quantization(arm), "gptq_marlin")

    def test_tree_arm_spec_has_token_tree(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "mtp3_tree")
        spec = arm_spec_config(arm)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["method"], "mtp")
        self.assertEqual(spec["num_speculative_tokens"], TREE_DEPTH)
        self.assertEqual(len(ast.literal_eval(spec["speculative_token_tree"])), 14)
        self.assertTrue(arm.requires_mtp)
        self.assertFalse(arm.requires_gptq)

    def test_gptq_mtp3_requires_both(self) -> None:
        arm = next(a for a in build_arms(self._cfg()) if a.name == "gptq_mtp3")
        self.assertTrue(arm.requires_mtp)
        self.assertTrue(arm.requires_gptq)
        self.assertEqual(arm.model, "/tmp/wemm-gptq")
        self.assertEqual(arm_quantization(arm), "gptq_marlin")
        spec = arm_spec_config(arm)
        self.assertEqual(spec["num_speculative_tokens"], 3)
        self.assertNotIn("speculative_token_tree", spec)

    def test_arms_dropped_when_prerequisites_missing(self) -> None:
        # gptq arms are always built (they need --gptq-model to run); MTP arms
        # need a sidecar, so they are absent here.
        arms = build_arms(RunConfig(), full=False)
        names = {a.name for a in arms}
        self.assertIn("gptq", names)  # requires_gptq, dropped by the orchestrator
        self.assertNotIn("mtp3", names)
        self.assertNotIn("gptq_mtp3", names)
        self.assertIn("baseline", names)
        self.assertIn("baseline_recheck", names)

    def test_select_arms_handles_new_names(self) -> None:
        arms = build_arms(self._cfg(), full=False)
        picked = select_arms(arms, "gptq,mtp3_tree,gptq_tree")
        self.assertEqual([a.name for a in picked], ["gptq", "mtp3_tree", "gptq_tree"])

    def test_arm_to_dict_exposes_spec_and_quantization(self) -> None:
        arm = Arm(
            name="x",
            label="x",
            kind="opt",
            extra_flags=["--quantization", "gptq_marlin", "--speculative-config", '{"method":"mtp"}'],
        )
        data = arm.to_dict()
        self.assertEqual(data["quantization"], "gptq_marlin")
        self.assertEqual(data["speculative"], {"method": "mtp"})


if __name__ == "__main__":
    unittest.main()
