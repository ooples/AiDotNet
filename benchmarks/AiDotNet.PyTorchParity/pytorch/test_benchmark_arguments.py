"""Argument checks that must fail before the benchmark prepares any workload.

Run with: python -m unittest discover -s benchmarks/AiDotNet.PyTorchParity/pytorch -p test_benchmark_arguments.py
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import benchmark
import benchmark_telemetry


def arguments(**overrides):
    values = dict(models="mlp", batch_size=64, output=Path("report.json"), hifigan_fixture=None,
                  hifigan_workload_receipt=None, lstm_fixture=None, lstm_workload_receipt=None, lstm_numerics=None,
                  lstm_cpu_mkldnn=None, require_matched_workload=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class BenchmarkArgumentTests(unittest.TestCase):
    def test_hifigan_rejects_other_training_batch_sizes_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "--batch-size 1"):
            benchmark.run(arguments(models="hifigan-v1", batch_size=64))

    def test_hifigan_receipt_requires_the_fixture(self):
        with self.assertRaisesRegex(ValueError, "requires --hifigan-fixture"):
            benchmark.run(arguments(hifigan_workload_receipt=Path("receipt.json")))

    def test_report_and_receipt_may_not_be_one_file(self):
        with tempfile.TemporaryDirectory(prefix="parity-arguments-") as temp:
            report = Path(temp) / "report.json"
            same = Path(temp) / "nested" / ".." / "report.json"
            for fixture, receipt in (("hifigan_fixture", "hifigan_workload_receipt"),
                                     ("lstm_fixture", "lstm_workload_receipt")):
                with self.subTest(fixture=fixture), self.assertRaisesRegex(ValueError, "resolve to the same file"):
                    benchmark._reject_artifact_collisions(
                        arguments(output=report, **{fixture: Path(temp) / "fixture", receipt: same}))

    def test_distinct_and_default_receipt_paths_are_accepted(self):
        with tempfile.TemporaryDirectory(prefix="parity-arguments-") as temp:
            report = Path(temp) / "report.json"
            benchmark._reject_artifact_collisions(arguments(output=report, hifigan_fixture=Path(temp) / "fixture"))
            benchmark._reject_artifact_collisions(arguments(output=report, lstm_fixture=Path(temp) / "fixture",
                                                            lstm_numerics=Path(temp) / "numerics"))
            benchmark._reject_artifact_collisions(arguments(output=report))


class PhaseMarkerTests(unittest.TestCase):
    def test_an_inactive_marker_reports_that_it_did_not_reset_the_peaks(self):
        import os
        previous = os.environ.pop("AIDOTNET_BENCH_PHASE_FILE", None)
        try:
            self.assertIs(benchmark_telemetry.mark_phase("mlp", "inference-b1-steady"), False)
        finally:
            if previous is not None:
                os.environ["AIDOTNET_BENCH_PHASE_FILE"] = previous

    def test_an_active_cpu_marker_writes_the_phase_and_does_not_claim_a_reset(self):
        import json
        import os
        with tempfile.TemporaryDirectory(prefix="parity-phase-") as temp:
            phases = Path(temp) / "phases.jsonl"
            saved = {key: os.environ.get(key) for key in ("AIDOTNET_BENCH_PHASE_FILE", "AIDOTNET_BENCH_DEVICE")}
            os.environ.update(AIDOTNET_BENCH_PHASE_FILE=str(phases), AIDOTNET_BENCH_DEVICE="cpu")
            try:
                self.assertIs(benchmark_telemetry.mark_phase("mlp", "inference-b1-steady"), False)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            record = json.loads(phases.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["phase"], "inference-b1-steady")


if __name__ == "__main__":
    unittest.main()
