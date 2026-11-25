"""Tests for run_matrix.py's per-row schedules, median/IQR summary, loss evidence and classical summary."""
import sys
import tempfile
import unittest
from pathlib import Path

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


class ClockTests(unittest.TestCase):
    def tearDown(self):
        rm.LOCKED_CLOCKS.clear()

    def test_clock_flags(self):
        steady = dict(loadedSamples=5, smMedianMhz=1500, smMinMhz=1495, smMaxMhz=1505, memMedianMhz=7000)
        self.assertEqual(rm.clock_flags(steady), set())
        self.assertEqual(rm.clock_flags(dict(steady, smMinMhz=1200)), {"clock-varied"})
        rm.LOCKED_CLOCKS["sm"] = 1800
        self.assertEqual(rm.clock_flags(steady), {"clock-unlocked"})
        self.assertEqual(rm.clock_flags(dict(loadedSamples=0)), set())


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
