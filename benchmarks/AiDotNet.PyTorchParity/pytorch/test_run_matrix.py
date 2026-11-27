"""Tests for run_matrix.py's per-row schedules, median/IQR summary, loss evidence and classical summary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_matrix as rm  # noqa: E402


def launch_rows(model, configuration, values, operation="train", batch=64, first=2.0, last=1.0, parameters=10):
    return [dict(model=model, device="cpu", launch=i + 1, configuration=configuration, operation=operation, batch=batch,
                 parameters=parameters, ms=v, first_loss=first, last_loss=last, sm_mhz=None, flags="")
            for i, v in enumerate(values)]


class QuantileTests(unittest.TestCase):
    def test_matches_numpy_linear_interpolation(self):
        values = [9, 1, 8, 2, 7, 3, 6, 4, 5]
        self.assertEqual(rm.quantile(values, 0.5), 5)
        self.assertEqual(rm.quantile(values, 0.25), 3)
        self.assertEqual(rm.quantile(values, 0.75), 7)
        self.assertAlmostEqual(rm.quantile([1, 2, 3, 4], 0.25), 1.75)


class ClockSampleTests(unittest.TestCase):
    def test_lines_cut_off_or_unreadable_are_discarded(self):
        class Exited:
            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "launch"
            stem.with_suffix(".clocks.csv").write_text(
                "1500, 7000, 90, 60, 200\n1500,7000\n[N/A], 7000, 90, 60, 200\n", encoding="utf-8")
            clocks = rm.stop_clock_sampler((Exited(), open(Path(directory) / "out.txt", "w")), stem)
        self.assertEqual((clocks["samples"], clocks["loadedSamples"], clocks["smMedianMhz"]), (1, 1, 1500.0))

    def test_unavailable_power_or_temperature_keeps_the_clock_sample(self):
        class Exited:
            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "launch"
            stem.with_suffix(".clocks.csv").write_text(
                "1500, 7000, 90, 60, [N/A]\n1510, 7000, 90, [N/A], [N/A]\n1500, [N/A], 90, 60, 200\n", encoding="utf-8")
            with open(Path(directory) / "out.txt", "w") as out:
                clocks = rm.stop_clock_sampler((Exited(), out), stem)
        self.assertEqual((clocks["samples"], clocks["loadedSamples"], clocks["smMedianMhz"], clocks["memMedianMhz"],
                          clocks["maxTemperatureC"]), (2, 2, 1505.0, 7000.0, 60.0))

    def test_no_sampling_without_the_benchmark_gpu(self):
        previous = rm.GPU_ID
        try:
            rm.GPU_ID = None
            with mock.patch.object(rm.shutil, "which", return_value="nvidia-smi"), \
                    mock.patch.object(rm.subprocess, "Popen") as popen:
                self.assertIsNone(rm.start_clock_sampler(Path("unused"), "cuda"))
            popen.assert_not_called()
        finally:
            rm.GPU_ID = previous

    def test_a_single_listed_gpu_identifies_the_benchmark_gpu(self):
        def listing(stdout):
            return mock.patch.object(rm.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=stdout))
        with mock.patch.object(rm.shutil, "which", return_value="nvidia-smi"):
            with listing("GPU-aaaa\n"):
                self.assertEqual(rm.sole_gpu_id(), "GPU-aaaa")
            with listing("GPU-aaaa\nGPU-bbbb\n"):
                self.assertIsNone(rm.sole_gpu_id())

    def test_queries_name_the_benchmark_gpu(self):
        previous = rm.GPU_ID
        try:
            rm.GPU_ID = None
            self.assertEqual(rm.gpu_selector(), [])
            rm.GPU_ID = "GPU-0000"
            self.assertEqual(rm.gpu_selector(), ["--id=GPU-0000"])
        finally:
            rm.GPU_ID = previous


class ScheduleTests(unittest.TestCase):
    def test_core_rows_keep_the_shared_schedule(self):
        epochs, batches, operations = rm.schedule(rm.row_arguments("resnet18", smoke=False))
        self.assertEqual((epochs, batches), (3, 20))
        self.assertEqual(operations, [("train", 64), ("inference", 1), ("inference", 8), ("inference", 16)])

    def test_family_rows_carry_their_own_shape(self):
        _, batches, operations = rm.schedule(rm.row_arguments("ddpm", smoke=False))
        self.assertEqual(batches, 5)
        self.assertEqual(operations, [("train", 4), ("inference", 1), ("inference", 4)])
        self.assertEqual(rm.schedule(rm.row_arguments("gat", smoke=True))[2], [("train", 1), ("inference", 1)])

    def test_old_runs_without_row_arguments_fall_back_to_common(self):
        info = dict(common=rm.row_arguments("mlp", smoke=True))
        self.assertEqual(rm.row_common(info, "cnn"), info["common"])

    def test_every_family_is_selectable_and_nine_launches_is_the_default(self):
        self.assertEqual(rm.GROUPS["families"], list(rm.FAMILIES))
        self.assertTrue(set(rm.FAMILIES) <= set(rm.MODELS))
        self.assertEqual(rm.DEFAULT_LAUNCHES, 9)


class SummaryTests(unittest.TestCase):
    def test_median_and_iqr_over_launches(self):
        ai = list(range(1, 10))
        torch = [2 * v for v in ai]
        [s] = rm.summarize(launch_rows("gcn", "aidotnet", ai) + launch_rows("gcn", "torch-compile", torch), 9)
        self.assertEqual((s["ai_median_ms"], s["ai_q1_ms"], s["ai_q3_ms"], s["ai_iqr_ms"]), (5, 3, 7, 4))
        self.assertEqual((s["torch_median_ms"], s["torch_iqr_ms"]), (10, 8))
        self.assertEqual((s["torch_over_ai"], s["launch_ratio_q1"], s["launch_ratio_q3"]), (2, 2, 2))
        self.assertEqual(s["launches"], 9)

    def test_flags_a_loss_that_does_not_fall(self):
        rows = launch_rows("gcn", "aidotnet", [1, 1, 1]) + launch_rows("gcn", "torch-compile", [1, 1, 1], first=1.0, last=1.5)
        [s] = rm.summarize(rows, 3)
        self.assertIn("loss-not-falling", s["flags"])
        self.assertEqual((s["torch_first_loss"], s["torch_last_loss"]), (1.0, 1.5))

    def test_flags_missing_launches_and_parameter_mismatch(self):
        rows = launch_rows("gcn", "aidotnet", [1, 1]) + launch_rows("gcn", "torch-compile", [1, 1], parameters=11)
        [s] = rm.summarize(rows, 9)
        self.assertEqual(s["flags"], "missing-launches;parameters-differ")

    def test_markdown_reports_median_iqr_parameters_and_losses(self):
        rows = launch_rows("ddpm", "aidotnet", [2, 2.5, 3], batch=4) + launch_rows("ddpm", "torch-compile", [4, 5, 6], batch=4)
        table = rm.markdown(rm.summarize(rows, 3))
        self.assertIn("| ddpm | train | 4 | 2.5 (2.25–2.75) | 5 (4.5–5.5) torch-compile | 2.00 (2.00–2.00) | 10 / 10 | 2 → 1; 2 → 1 |",
                      table)


def completed_process(returncode, output=""):
    return SimpleNamespace(returncode=returncode, stdout=output, stderr="")


class ClockTests(unittest.TestCase):
    def tearDown(self):
        rm.REQUESTED_CLOCKS.clear()

    def test_clock_flags(self):
        steady = dict(loadedSamples=5, smMedianMhz=1500, smMinMhz=1495, smMaxMhz=1505, memMedianMhz=7000)
        self.assertEqual(rm.clock_flags(steady), set())
        self.assertEqual(rm.clock_flags(dict(steady, smMinMhz=1200)), {"clock-varied"})
        rm.REQUESTED_CLOCKS["sm"] = 1800
        self.assertEqual(rm.clock_flags(steady), {"clock-unlocked"})
        self.assertEqual(rm.clock_flags(dict(loadedSamples=0)), set())

    def test_a_refused_lock_still_checks_the_requested_clock(self):
        steady = dict(loadedSamples=5, smMedianMhz=1500, smMinMhz=1495, smMaxMhz=1505, memMedianMhz=7000)
        result = {}
        with mock.patch.object(rm.subprocess, "run", return_value=completed_process(1, "refused")), \
                mock.patch("builtins.print"):
            rm.lock_clocks(1800, None, result)
        self.assertEqual(result["sm"]["exitCode"], 1)
        rm.REQUESTED_CLOCKS["sm"] = 1800
        self.assertEqual(rm.clock_flags(steady), {"clock-unlocked"})
        rm.REQUESTED_CLOCKS["sm"] = 1510  # refused, but the GPU ran within tolerance: the refusal alone is no flag
        self.assertEqual(rm.clock_flags(steady), set())

    def test_reset_only_the_locks_that_were_taken(self):
        locked = dict(sm=dict(mhz=1800, exitCode=0, output=""), mem=dict(mhz=7001, exitCode=1, output="refused"))
        with mock.patch.object(rm.subprocess, "run", return_value=completed_process(0)) as run:
            results, failures = rm.reset_clocks(locked)
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ["-rgc"])
        self.assertEqual((list(results), failures), (["sm"], []))

    def test_a_failed_reset_is_reported_after_every_reset_is_attempted(self):
        locked = dict(sm=dict(mhz=1800, exitCode=0, output=""), mem=dict(mhz=7001, exitCode=0, output=""))
        outcomes = iter([completed_process(6, "no permission"), completed_process(0)])
        with mock.patch.object(rm.subprocess, "run", side_effect=lambda *a, **k: next(outcomes)) as run:
            results, failures = rm.reset_clocks(locked)
        self.assertEqual(run.call_count, 2)
        self.assertEqual((results["sm"]["exitCode"], results["mem"]["exitCode"]), (6, 0))
        self.assertEqual(len(failures), 1)
        self.assertIn("-rgc exited 6", failures[0])

    def test_clock_lock_file_is_per_gpu(self):
        self.assertEqual(rm.clock_lock_path("GPU-ab:cd").name, ".run_matrix.clocks-GPU-ab_cd.lock")
        self.assertNotEqual(rm.clock_lock_path("GPU-1"), rm.clock_lock_path("GPU-2"))


class ClockRequestTests(unittest.TestCase):
    def test_a_new_run_uses_the_flags(self):
        self.assertEqual(rm.resolve_clock_request({}, 1800, None, resuming=False), (1800, None))

    def test_a_resume_reuses_the_recorded_request(self):
        info = dict(clockLockRequest=dict(sm=1800, mem=7001))
        self.assertEqual(rm.resolve_clock_request(info, None, None, resuming=True), (1800, 7001))
        self.assertEqual(rm.resolve_clock_request(info, 1800, 7001, resuming=True), (1800, 7001))
        with self.assertRaises(ValueError):
            rm.resolve_clock_request(info, 1700, None, resuming=True)

    def test_a_resume_cannot_add_locks_to_an_unlocked_run(self):
        for info in (dict(clockLockRequest=dict(sm=None, mem=None)), {}):
            self.assertEqual(rm.resolve_clock_request(info, None, None, resuming=True), (None, None))
            with self.assertRaises(ValueError):
                rm.resolve_clock_request(info, 1800, None, resuming=True)

    def test_runs_recorded_before_the_request_field_use_their_first_lock_attempt(self):
        info = dict(clockLocks=[dict(time="t", sm=dict(mhz=1800, exitCode=1, output="refused"))])
        self.assertEqual(rm.resolve_clock_request(info, None, None, resuming=True), (1800, None))


class ReportValidationTests(unittest.TestCase):
    def test_an_aidotnet_report_at_another_training_batch_is_rejected(self):
        common = rm.row_arguments("mlp", smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for protocol in (dict(Precision="fp32", BatchSize=32), dict(Precision="fp32")):
                path.write_text(json.dumps(dict(Framework="AiDotNet", Results=[{}], Protocol=protocol)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "trained at batch"):
                    rm.measurements(path, "aidotnet", "mlp", "cpu", common)

    def test_a_pytorch_report_at_another_training_batch_is_rejected(self):
        common = rm.row_arguments("mlp", smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(dict(framework="PyTorch", results=[{}],
                                            protocol=dict(precision="fp32", batch_size=32))), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trained at batch 32, expected 64"):
                rm.measurements(path, "torch-compile", "mlp", "cpu", common)


class ClassicalResumeTests(unittest.TestCase):
    @staticmethod
    def receipt(stem, code=0):
        stem.with_suffix(".receipt.json").write_text(json.dumps(dict(exitCode=code)), encoding="utf-8")

    @staticmethod
    def resume(dest, launches):
        """Run run_classical over DEST with run() replaced; return (process, files in its launch directory) per run."""
        info = dict(launches=launches, classicalRepeats=1)
        (dest / "protocol.json").write_text(json.dumps(info), encoding="utf-8")
        ran = []

        def fake_run(cmd, stem, env, device=None, allow_battery=False):
            directory = dest / f"classical-{stem.name.rsplit('-', 1)[1]}"
            ran.append((stem.name, sorted(p.name for p in directory.iterdir())))
            return []

        with mock.patch.object(rm, "run", side_effect=fake_run), mock.patch("builtins.print"):
            rm.run_classical(dest, info, {}, allow_battery=True)
        return ran

    def test_a_side_with_a_receipt_but_no_report_runs_again(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            launch = dest / "classical-1"
            launch.mkdir()
            for name in ("train.csv", "test.csv", "aidotnet_classical.json"):
                (launch / name).write_text("x", encoding="utf-8")
            self.receipt(dest / "classical-aidotnet-1")
            self.receipt(dest / "classical-python-1")  # exited 0, but python_classical.json is missing
            ran = self.resume(dest, 1)
        self.assertEqual([name for name, _ in ran], ["classical-python-1"])

    def test_a_partial_seed_is_repaired_before_python_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for launch in (1, 2):
                (dest / f"classical-{launch}").mkdir()
            for name in ("train.csv", "test.csv", "aidotnet_classical.json", "python_classical.json"):
                (dest / "classical-1" / name).write_text("x", encoding="utf-8")
            for side in ("aidotnet", "python"):
                self.receipt(dest / f"classical-{side}-1")
            (dest / "classical-2" / "train.csv").write_text("x", encoding="utf-8")  # the copy stopped before test.csv
            ran = self.resume(dest, 2)
        self.assertEqual(ran[0], ("classical-python-2", ["test.csv", "train.csv"]))


class ClassicalTests(unittest.TestCase):
    def test_summary_compares_against_the_fastest_library(self):
        rows = []
        for launch in (1, 2, 3):
            rows += [dict(row="hgb", library="AiDotNet", launch=launch, fit_s=1.0, predict_s=0.1, metric="accuracy",
                          value=0.8, flags=""),
                     dict(row="hgb", library="scikit-learn", launch=launch, fit_s=2.0, predict_s=0.1, metric="accuracy",
                          value=0.8, flags=""),
                     dict(row="hgb", library="lightgbm", launch=launch, fit_s=1.5, predict_s=0.1, metric="accuracy",
                          value=0.81, flags="")]
        [entry] = rm.summarize_classical(rows, 3)
        self.assertEqual((entry["fastest_rival"], entry["rival_over_ai_fit"]), ("lightgbm", 1.5))
        self.assertEqual(entry["flags"], "")
        self.assertIn("| hgb | 1,000 (1,000–1,000) | 2,000 (2,000–2,000) | – | 1,500 (1,500–1,500) | 1.50 (lightgbm) |",
                      rm.classical_markdown([entry]))


if __name__ == "__main__":
    unittest.main()
