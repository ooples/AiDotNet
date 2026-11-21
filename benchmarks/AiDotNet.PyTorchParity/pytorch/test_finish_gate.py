import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import psutil

from benchmark_telemetry import cpu_storage_events
from finish_gate import cache_environment, contending_processes, correctness_gate, evaluate, gpu_allocator_peaks, hifigan_comparison_kinds, hifigan_comparison_policy, hifigan_required_checks, launch, memory_metrics, metrics, normalize, paired_gate, phase_group, reference_recipe_gate


class GateTests(unittest.TestCase):
    def test_overflowing_derived_ratios_stay_json_serializable(self):
        result = paired_gate([1.0] * 5, [1e-320] * 5)
        self.assertEqual("unverified", result["status"])
        json.dumps(result, allow_nan=False)

    def test_evaluate_gates_memory_at_parity_and_timing_at_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = [dict(model="tiny", device="cpu", mode="score", pair=pair, side=side, path=temporary)
                    for pair in range(5) for side in ("aidotnet", "pytorch")]
            def measured(run):
                value = .95 if run["side"] == "aidotnet" else 1.0
                return {"whole_process_rss_bytes": value, "training_seconds": value}, []
            with patch("finish_gate.metrics", side_effect=measured), patch("finish_gate.correctness_gate", return_value={"status": "pass"}):
                outcomes = evaluate(runs, .9)["tiny/cpu"]["metrics"]
            self.assertEqual("pass", outcomes["whole_process_rss_bytes"]["status"])
            self.assertEqual("fail", outcomes["training_seconds"]["status"])

    def launch_config(self):
        return {"commands": {side: {"argv": ["runner"], "cwd": ".", "env": {"CUDA_VISIBLE_DEVICES": "wrong"}}
                             for side in ("aidotnet", "pytorch")}, "protocol": {"inference_batch_sizes": [1]},
                "cache_policy": {"regime": "cold"}, "gpu_id": "2"}

    def test_cuda_launch_pins_both_runners_and_sampler_to_physical_uuid(self):
        config = self.launch_config()
        with tempfile.TemporaryDirectory() as temporary:
            for side in ("aidotnet", "pytorch"):
                sampler = Mock(contention=[])
                sampler.finish.return_value = []
                with patch("finish_gate.provenance", return_value={}), patch("finish_gate.contending_processes", return_value=[]), \
                     patch("finish_gate.nvidia_query", return_value="2, GPU-test, 00000000:65:00.0") as query, \
                     patch("finish_gate.subprocess.Popen", return_value=Mock(pid=123, wait=Mock(return_value=0))) as popen, \
                     patch("finish_gate.Sampler", return_value=sampler) as sampling:
                    result = launch(config, side, "tiny", "cuda", 0, "score", Path(temporary))
                env = popen.call_args.kwargs["env"]
                self.assertEqual("GPU-test", env["CUDA_VISIBLE_DEVICES"])
                self.assertEqual("PCI_BUS_ID", env["CUDA_DEVICE_ORDER"])
                self.assertEqual("cuda", env["AIDOTNET_DIRECTGPU_BACKENDS"])
                sampling.assert_called_once_with(123, "GPU-test")
                query.assert_called_once_with("2", "index,uuid,pci.bus_id")
                saved = json.loads((Path(result["path"]) / "measurement.json").read_text())
                self.assertEqual("2", saved["gpu_mapping"]["physical_index"])
                self.assertEqual("GPU-test", saved["environment_overrides"]["CUDA_VISIBLE_DEVICES"])

    def test_cuda_launch_without_resolved_physical_gpu_fails_before_start(self):
        with tempfile.TemporaryDirectory() as temporary, patch("finish_gate.nvidia_query", return_value=None), \
             patch("finish_gate.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "physical nvidia-smi GPU"):
                launch(self.launch_config(), "aidotnet", "tiny", "cuda", 0, "score", Path(temporary))
            popen.assert_not_called()

    def test_exited_child_writes_an_invalid_measurement(self):
        with tempfile.TemporaryDirectory() as temporary, patch("finish_gate.provenance", return_value={}), \
             patch("finish_gate.contending_processes", return_value=[]), \
             patch("finish_gate.subprocess.Popen", return_value=Mock(pid=123, wait=Mock(return_value=7))), \
             patch("finish_gate.Sampler", side_effect=psutil.NoSuchProcess(123)):
            result = launch(self.launch_config(), "aidotnet", "tiny", "cpu", 0, "score", Path(temporary))
            saved = json.loads((Path(result["path"]) / "measurement.json").read_text())
            self.assertEqual(7, saved["return_code"])
            self.assertEqual({}, saved["memory"])
            self.assertIn("before memory sampling", saved["invalid_reason"])
            self.assertEqual(({}, [saved["invalid_reason"]]), metrics(dict(saved, path=result["path"])))

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

    def test_allocator_peaks_belong_to_previous_interval_and_missing_or_multiple_devices_stay_unknown(self):
        phases = [{"gpu_memory_interval": {"previous_phase": "construction", "coverage": "owning_backend_buffers_and_registered_driver_pools",
                   "devices": {"cuda:0": {"PeakAllocatedBytes": 100, "PeakReservedBytes": 400}}}},
                  {"gpu_memory_interval": {"previous_phase": "inference-b16-steady", "coverage": "torch_cuda_caching_allocator",
                   "devices": {"0": {"peak_allocated_bytes": 150, "peak_reserved_bytes": 800}}}}]
        result = gpu_allocator_peaks(phases)
        self.assertEqual(400, result["construction_gpu_allocator_peak_reserved_bytes"])
        self.assertEqual(150, result["inference_b16_gpu_allocator_peak_allocated_bytes"])
        self.assertEqual(800, result["whole_process_gpu_allocator_peak_reserved_bytes"])
        self.assertNotIn("training_gpu_allocator_peak_reserved_bytes", result)
        phases[0]["gpu_memory_interval"]["devices"]["cuda:1"] = {"PeakAllocatedBytes": 900}
        self.assertNotIn("construction_gpu_allocator_peak_allocated_bytes", gpu_allocator_peaks(phases))
        self.assertEqual({}, gpu_allocator_peaks([{"gpu_memory_interval": {"coverage": "unavailable"}}]))

    def test_reference_recipe_must_match_scored_source_and_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            hashes = {"pytorch_benchmark": "a" * 64}
            run = {"artifacts": hashes, "artifact_paths": {"hifigan_reference_manifest": str(path)}}
            def bind(source, exporter=None):
                manifest = {"benchmark_source_sha256": source}
                if exporter is not None:
                    manifest["exporter_source_sha256"] = exporter
                path.write_text(json.dumps(manifest))
                hashes["hifigan_reference_manifest"] = hashlib.sha256(path.read_bytes()).hexdigest()
            bind(hashes["pytorch_benchmark"])
            self.assertEqual("pass", reference_recipe_gate(run)["status"])
            # Even a correctly rehashed manifest must not validate a different training recipe.
            bind("b" * 64)
            self.assertIn("recipe does not match", reference_recipe_gate(run)["reason"])
            bind(hashes["pytorch_benchmark"])
            path.write_text("changed")
            self.assertEqual("unverified", reference_recipe_gate(run)["status"])
            bind(None)
            self.assertEqual("unverified", reference_recipe_gate(run)["status"])
            bind("a" * 64)
            hashes["reference_exporter"] = "c" * 64
            bind("a" * 64, exporter="d" * 64)
            self.assertIn("exporter hash does not match", reference_recipe_gate(run)["reason"])
            bind("a" * 64, exporter=hashes["reference_exporter"])
            self.assertEqual("pass", reference_recipe_gate(run)["status"])
            hashes.pop("pytorch_benchmark")
            self.assertEqual("unverified", reference_recipe_gate(run)["status"])

    def test_equivalence_receipt_must_match_binaries_reference_and_fused_updates(self):
        hashes = {"AiDotNet": "a" * 64, "Tensors": "b" * 64, "hifigan_reference_manifest": "c" * 64}
        required, learned, common = hifigan_required_checks()
        self.assertEqual((1615, 388, 404), (len(required), len(learned), len(common)))
        kinds = hifigan_comparison_kinds()
        self.assertEqual(1164, sum(kind == "learned_state" for kind in kinds.values()))
        self.assertEqual(48, sum(kind == "spectral_state" for kind in kinds.values()))
        self.assertEqual(451, sum(kind != "learned_state" for kind in kinds.values()))
        checks = {name: {"name": name, "pass": True, "finite": True, "elements": 1,
                         "comparison_kind": kinds[name], "max_absolute_error": 0, "relative_l2": 0,
                         "cosine": 1, "norm_ratio": 1, "reference_max_absolute_value": 1} for name in required}
        first_learned, first_common = min(learned), min(common)
        checks["isolated-gradient/" + first_learned]["elements"] += 84_660_721 - len(learned)
        for step in (1, 2, 3):
            checks[f"train/step{step}/" + first_common]["elements"] += 84_679_257 - len(common)
        for batch in (1, 8, 16):
            checks[f"inference/output/{batch}"]["elements"] = batch * 65_536
        checks["generator/output0"]["elements"] = 8192
        receipt = {"schema": "hifigan-v1-equivalence-receipt-v1", "model": "hifigan-v1", "device": "cuda",
                   "pass": True, "checks": list(checks.values()),
                   "errors": [], "fused_steps": 6, "eager_steps": 0, "assemblies_sha256": {key: hashes[key] for key in ("AiDotNet", "Tensors")},
                   "tolerance": {"absolute": 2e-4, "relative": 3e-3, "gradient_relative": .03, "gradient_cosine": .995},
                   "reference_manifest_sha256": hashes["hifigan_reference_manifest"],
                   "comparison_policy": hifigan_comparison_policy("cuda")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "equivalence.json"
            manifest_path = Path(temporary) / "manifest.json"
            hashes["pytorch_benchmark"] = "e" * 64
            manifest_path.write_text(json.dumps({"benchmark_source_sha256": hashes["pytorch_benchmark"]}))
            hashes["hifigan_reference_manifest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            receipt["reference_manifest_sha256"] = hashes["hifigan_reference_manifest"]
            run = {"model": "hifigan-v1", "device": "cuda", "artifacts": hashes,
                   "artifact_paths": {"hifigan_reference_manifest": str(manifest_path)}}
            def bind(value):
                path.write_text(json.dumps(value))
                run["correctness_receipt"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            bind(receipt)
            self.assertEqual("pass", correctness_gate(run)["status"])
            bind(dict(receipt, eager_steps=1))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, tolerance={"absolute": 1}))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, reference_manifest_sha256="d" * 64))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, checks=[]))
            self.assertEqual("fail", correctness_gate(run)["status"])
            bind(dict(receipt, checks=list(checks.values())[:-1]))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, checks=[*checks.values(), next(iter(checks.values()))]))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, **{"pass": False}))
            self.assertEqual("fail", correctness_gate(run)["status"])
            legacy = dict(receipt)
            legacy.pop("comparison_policy")
            bind(legacy)
            self.assertEqual("unverified", correctness_gate(run)["status"])
            bind(dict(receipt, comparison_policy=dict(receipt["comparison_policy"], learned_state_checks=1163)))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            policy = dict(receipt["comparison_policy"], learned_state=dict(receipt["comparison_policy"]["learned_state"], cosine_min=.1))
            bind(dict(receipt, comparison_policy=policy))
            self.assertEqual("unverified", correctness_gate(run)["status"])
            def altered(name, **fields):
                return [dict(row, **fields) if row["name"] == name else row for row in checks.values()]
            learned_state = "train/step1/" + first_learned
            for device, absolute, relative, gradient_relative, cosine in (("cpu", 2e-5, 1e-3, .01, .999), ("cuda", 2e-4, 3e-3, .03, .995)):
                run["device"] = device
                current = dict(receipt, device=device,
                               tolerance={"absolute": absolute, "relative": relative,
                                          "gradient_relative": gradient_relative, "gradient_cosine": cosine},
                               comparison_policy=hifigan_comparison_policy(device))
                bind(dict(current, checks=altered(learned_state, max_absolute_error=100)))
                self.assertEqual("pass", correctness_gate(run)["status"])
                bind(dict(current, checks=altered(learned_state, relative_l2=relative * 1.01)))
                self.assertEqual("fail", correctness_gate(run)["status"])
                bind(dict(current, checks=altered(learned_state, cosine=cosine - 1e-5)))
                self.assertEqual("fail", correctness_gate(run)["status"])
                bind(dict(current, checks=altered(learned_state, finite=False)))
                self.assertEqual("fail", correctness_gate(run)["status"])
                spectral_state = "train/step1/" + min(common - learned)
                for original_check in ("isolated-gradient/" + first_learned, spectral_state, "generator/output0"):
                    bind(dict(current, checks=altered(original_check, max_absolute_error=1, relative_l2=1)))
                    self.assertEqual("fail", correctness_gate(run)["status"])
                bind(dict(current, checks=altered("isolated-gradient-summary/generator", norm_ratio=1 + 2 * gradient_relative)))
                self.assertEqual("fail", correctness_gate(run)["status"])
                bind(dict(current, checks=altered(spectral_state, comparison_kind="learned_state")))
                self.assertEqual("unverified", correctness_gate(run)["status"])
            run["device"] = "cuda"
            bind(receipt)
            path.write_text("changed")
            self.assertEqual("unverified", correctness_gate(run)["status"])


if __name__ == "__main__":
    unittest.main()
