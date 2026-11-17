import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_telemetry import cpu_storage_events
from finish_gate import cache_environment, contending_processes, evaluate, memory_metrics, metrics, normalize, paired_gate, phase_group


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

    def test_cold_cache_paths_are_private_to_each_fresh_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first.mkdir()
            second.mkdir()
            a, state = cache_environment(first, {"regime": "cold"}, "pytorch")
            b, _ = cache_environment(second, {"regime": "cold"}, "pytorch")
            self.assertEqual("known_environment_redirects_only", state["coverage"])
            self.assertTrue(set(a.values()).isdisjoint(b.values()))
            self.assertIn("CUDA_CACHE_PATH", a)
            self.assertIn("TORCHINDUCTOR_CACHE_DIR", a)
            self.assertTrue(all(Path(path).is_dir() for path in a.values()))

    def test_changed_or_empty_warm_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed"
            run = root / "run"
            seed.mkdir()
            run.mkdir()
            (seed / "kernel").write_bytes(b"changed")
            policy = {"regime": "warm", "warm_seeds": {"pytorch": {
                "path": str(seed), "sha256": {"kernel": "incorrect"}}}}
            with self.assertRaisesRegex(ValueError, "differs"):
                cache_environment(run, policy, "pytorch")

    def test_missing_reports_cannot_pass_or_silently_drop_allocator_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = {"path": temporary, "model": "hifigan", "device": "cuda", "mode": "score",
                   "pair": 0, "side": "aidotnet", "return_code": 1}
            result = evaluate([run], .9)["hifigan/cuda"]
        self.assertEqual("unfinished", result["status"])
        self.assertIn("missing pair member", result["protocol_errors"])
        for metric in ("cpu_framework_requested_bytes", "cpu_backing_count", "gpu_framework_requested_count",
                       "gpu_backing_bytes", "first_epoch_cache_coverage"):
            self.assertEqual("unverified", result["metrics"][metric]["status"])

    def test_omitted_required_model_device_is_unfinished(self):
        result = evaluate([], .9, [("hifigan", "cpu"), ("hifigan", "cuda")])
        self.assertEqual({"hifigan/cpu", "hifigan/cuda"}, set(result))
        for row in result.values():
            self.assertEqual("unfinished", row["status"])
            self.assertIn("required model/device has no scored runs", row["protocol_errors"])

    def test_protocol_mismatch_and_missing_phase_memory_are_not_hidden(self):
        protocol = {"epochs": 2, "train_batches": 2, "batch_size": 4, "inference_iterations": 2,
                    "warmup_iterations": 1, "warmup_min_ms": 0, "seed": 1234,
                    "inference_batch_sizes": [1, 8, 16], "threads": 8}
        report = {"protocol": dict(protocol, compile=True, warmup_iterations=2), "torch_num_threads": 8,
                  "results": [{"model": "hifigan", "device": "cuda", "parameters": 10,
                    "training": {"epoch_seconds": [10, 8], "steady_state_epoch_seconds_median": 8,
                                 "first_loss": 1, "last_loss": .9}, "inference": []}]}
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "report.json").write_text(json.dumps(report))
            run = {"path": temporary, "return_code": 0, "side": "pytorch", "model": "hifigan",
                   "device": "cuda", "protocol": protocol, "memory": {"whole_process_rss_bytes": 100},
                   "mode": "score", "contention": [{"processes": [{"pid": 123, "name": "Runner.Worker.exe"}]}]}
            values, errors = metrics(run)
        self.assertIn("protocol mismatch: warmup_iterations", errors)
        self.assertIn("external CI worker/testhost contention observed during scoring", errors)
        self.assertIsNone(values["construction_rss_bytes"])
        self.assertIsNone(values["inference_b16_gpu_device_bytes"])

    def test_ci_worker_and_external_testhost_are_blockers_but_idle_listener_and_own_children_are_not(self):
        class Process:
            def __init__(self, pid, name, arguments=()):
                self.pid, self.arguments = pid, arguments
                self.info = {"pid": pid, "name": name, "create_time": 100}
            def cmdline(self):
                return self.arguments
        processes = [Process(1, "Runner.Worker.exe"), Process(2, "testhost.exe"),
                     Process(3, "Runner.Listener.exe"), Process(4, "dotnet", ["dotnet", "testhost.dll"]),
                     Process(5, "dotnet", ["dotnet", "AiDotNet.PyTorchParity.dll"])]
        with patch("finish_gate.psutil.process_iter", return_value=processes):
            blockers = contending_processes({2})
        self.assertEqual({1, 4}, {process["pid"] for process in blockers})
        self.assertTrue(all("arguments" not in process for process in blockers))


if __name__ == "__main__":
    unittest.main()
