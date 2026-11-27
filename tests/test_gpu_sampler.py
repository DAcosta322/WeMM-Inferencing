"""Regression test: GpuSampler is a Thread subclass and must not shadow _stop."""

from __future__ import annotations

import time
import unittest

from bench.server import GpuSampler


class GpuSamplerTest(unittest.TestCase):
    def test_start_stop_never_raises(self) -> None:
        sampler = GpuSampler(interval=0.05)
        sampler.start()
        time.sleep(0.2)
        result = sampler.stop()
        self.assertIn("samples", result)
        self.assertFalse(sampler.is_alive())

    def test_stop_twice_is_safe(self) -> None:
        sampler = GpuSampler(interval=0.05)
        sampler.start()
        sampler.stop()
        result = sampler.stop()
        self.assertIn("samples", result)


if __name__ == "__main__":
    unittest.main()
