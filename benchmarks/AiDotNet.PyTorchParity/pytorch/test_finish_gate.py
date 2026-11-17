import json
from pathlib import Path
import tempfile
import unittest

from benchmark_telemetry import cpu_storage_events
from finish_gate import memory_metrics, normalize, paired_gate, phase_group


class GateTests(unittest.TestCase):
    def test_repeatable_ten_percent_win_passes(self):
        result = paired_gate([8.8, 8.7, 8.8, 8.9, 8.8], [10] * 5)
        self.assertEqual("pass", result["status"])
        self.assertLessEqual(result["upper_95_ratio"], .9)

    def test_partial_speedup_fails(self):
        self.assertEqual("fail", paired_gate([9.5] * 5, [10] * 5)["status"])

    def test_noisy_point_win_does_not_hide_losing_pairs(self):
        result = paired_gate([8, 8, 8, 12, 12], [10] * 5)
        self.assertEqual("fail", result["status"])
        self.assertGreater(result["upper_95_ratio"], .9)

    def test_missing_or_nonfinite_values_never_pass(self):
        for values in ([8] * 4, [8, 8, 8, 8, None], [8, 8, 8, 8, float("nan")]):
            self.assertEqual("unverified", paired_gate(values, [10] * len(values))["status"])

    def test_zero_allocation_reference_is_a_tie_or_loss_not_percentage_win(self):
        self.assertEqual("fail", paired_gate([0] * 5, [0] * 5)["status"])
        self.assertEqual("pass", paired_gate([0] * 5, [1] * 5)["status"])

    def test_memory_peaks_include_construction_and_inference(self):
        phases = [{"unix_ms": 10, "phase": "construction"}, {"unix_ms": 20, "phase": "training-epoch-0"},
                  {"unix_ms": 30, "phase": "inference-b16-steady"}]
        samples = [{"unix_ms": 12, "rss_bytes": 100, "os_rss_hwm_bytes": 150},
                   {"unix_ms": 22, "rss_bytes": 90, "gpu_device_bytes": 400},
                   {"unix_ms": 32, "rss_bytes": 300, "gpu_device_bytes": 800}]
        result = memory_metrics(samples, phases)
        self.assertEqual(100, result["construction_rss_bytes"])
        self.assertEqual(300, result["whole_process_rss_bytes"])
        self.assertEqual(800, result["inference_b16_gpu_device_bytes"])
        self.assertIsNone(result["construction_gpu_device_bytes"])
        self.assertEqual("inference_b16", phase_group("inference-b16-complete"))

    def test_normalization_preserves_snake_case_and_reads_csharp(self):
        self.assertEqual({"steady_state_epoch_seconds_median": 1}, normalize({"SteadyStateEpochSecondsMedian": 1}))
        self.assertEqual({"cpu_threads": 8}, normalize({"cpu_threads": 8}))

    def test_cpu_allocator_events_are_counted_once_not_from_parent_operators(self):
        events = [{"name": "aten::add", "args": {"cpu_memory_usage": 4096}},
                  {"name": "[memory]", "args": {"Device Type": 0, "Bytes": 4096}},
                  {"name": "[memory]", "args": {"Device Type": 0, "Bytes": -4096}},
                  {"name": "[memory]", "args": {"Device Type": 1, "Bytes": 2048}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            result = cpu_storage_events(path)
        self.assertEqual(4096, result["bytes"])
        self.assertEqual(1, result["count"])


if __name__ == "__main__":
    unittest.main()
