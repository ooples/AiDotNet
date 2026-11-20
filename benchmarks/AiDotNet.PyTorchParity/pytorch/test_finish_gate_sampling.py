"""Sampling rules of the parity gate that need no benchmark process.

Run with: python -m unittest discover -s benchmarks/AiDotNet.PyTorchParity/pytorch -p test_finish_gate_sampling.py
"""

import math
import unittest

import finish_gate


class SamplingTests(unittest.TestCase):
    def test_median_of_huge_finite_ratios_stays_finite(self):
        self.assertEqual(finish_gate.midpoint_median([1e308] * 6), 1e308)
        self.assertTrue(math.isfinite(finish_gate.midpoint_median([1e308, 1.7e308])))
        self.assertEqual(finish_gate.midpoint_median([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(finish_gate.midpoint_median([4.0, 1.0, 2.0, 3.0]), 2.5)

    def test_a_gpu_query_that_crosses_a_phase_boundary_counts_only_toward_the_whole_process_peak(self):
        phases = [{"unix_ms": 100, "phase": "training-prepare"}, {"unix_ms": 200, "phase": "inference-b1-steady"}]
        inference = finish_gate.phase_group("inference-b1-steady")
        samples = [
            # Read during training, returned during inference: its observation phase is unknown.
            {"query_start_unix_ms": 150, "unix_ms": 250, "gpu_device_bytes": 9.0},
            {"query_start_unix_ms": 210, "unix_ms": 260, "gpu_device_bytes": 3.0},
            {"unix_ms": 220, "rss_bytes": 5.0},
        ]
        metrics = finish_gate.memory_metrics(samples, phases)
        self.assertEqual(metrics["whole_process_gpu_device_bytes"], 9.0)
        self.assertEqual(metrics[inference + "_gpu_device_bytes"], 3.0)
        self.assertEqual(metrics[inference + "_rss_bytes"], 5.0)

    def test_a_query_inside_one_phase_is_not_a_boundary_crossing(self):
        self.assertFalse(finish_gate.crosses_phase_boundary({"query_start_unix_ms": 210, "unix_ms": 260}, [100, 200]))
        self.assertTrue(finish_gate.crosses_phase_boundary({"query_start_unix_ms": 150, "unix_ms": 200}, [100, 200]))
        self.assertFalse(finish_gate.crosses_phase_boundary({"unix_ms": 260}, [100, 200]))


if __name__ == "__main__":
    unittest.main()
