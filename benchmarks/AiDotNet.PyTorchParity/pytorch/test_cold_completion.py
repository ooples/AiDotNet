import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finish_gate import cold_completion_seconds, evaluate, metrics


class ColdCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, epochs=3, mode="score", side="aidotnet"):
        protocol = {"epochs": epochs, "train_batches": 3, "batch_size": 1,
                    "inference_iterations": 5, "warmup_iterations": 10, "warmup_min_ms": 1000,
                    "seed": 1234, "inference_batch_sizes": [1], "threads": 8}
        names = ["construction", "construction-complete", "training-prepare"]
        for epoch in range(epochs):
            names += [f"training-epoch-{epoch}", f"training-epoch-{epoch}-complete"]
        names += ["training-complete", "inference-b1-prepare", "inference-b1-warmup",
                  "inference-b1-steady", "inference-b1-complete", "complete"]
        phases = [{"phase": name, "pid": 4242, "model": "fixture", "diagnostics": mode == "diagnostic",
                   "unix_ms": 1001000 + index * 1000} for index, name in enumerate(names)]
        run = {"path": str(self.root), "pid": 4242, "process_create_time": 1000.0,
               "model": "fixture", "device": "cpu", "side": side, "mode": mode,
               "return_code": 0, "protocol": protocol, "memory": {}, "diagnostic_files": {}}
        self.write_phases(run, phases)
        report = {"protocol": dict(protocol, compile=True), "torch_num_threads": 8,
                  "environment": {"cpu_threads": 8}, "results": [{"model": "fixture", "device": "cpu",
                      "training": {"epoch_seconds": [1] * epochs, "steady_state_epoch_seconds_median": 1,
                                   "first_loss": 2, "last_loss": 1}, "inference": []}]}
        (self.root / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return run, phases

    def write_phases(self, run, phases):
        path = self.root / "phases.jsonl"
        path.write_text("\n".join(map(json.dumps, phases)) + "\n", encoding="utf-8")
        run["diagnostic_files"]["phases"] = {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_constructor_and_preparation_are_included_and_epoch_phase_timer_is_preserved(self):
        run, _ = self.fixture()
        values, errors = metrics(run)
        self.assertEqual(1, values["first_epoch_seconds"])
        self.assertEqual(5, values["launch_to_first_epoch_end_seconds"])
        self.assertEqual([], errors)

    def test_official_epoch0_complete_is_used_for_single_epoch_and_both_sides(self):
        for side in ("aidotnet", "pytorch"):
            for mode in ("score", "diagnostic"):
                with self.subTest(side=side, mode=mode):
                    run, _ = self.fixture(epochs=1, mode=mode, side=side)
                    self.assertEqual(5, cold_completion_seconds(run))

    def test_missing_or_nonfinite_process_birth_is_unknown_not_an_epoch_only_win(self):
        run, _ = self.fixture()
        for created in (None, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(created=created):
                invalid = dict(run, process_create_time=created)
                with self.assertRaises(ValueError):
                    cold_completion_seconds(invalid)
                values, errors = metrics(invalid)
                self.assertIsNone(values["launch_to_first_epoch_end_seconds"])
                self.assertTrue(any("cold completion" in error for error in errors))
                self.assertEqual(1, values["first_epoch_seconds"])

    def test_phase_bounds_missing_endpoint_and_clock_changes_are_rejected(self):
        run, phases = self.fixture()
        invalid_sequences = [phases[:4] + phases[5:],
                             [dict(phases[0], unix_ms=999999)] + phases[1:],
                             phases[:4] + [dict(phases[4], unix_ms=1003000)] + phases[5:],
                             phases[:4] + [dict(phases[4], unix_ms=float("nan"))] + phases[5:]]
        for invalid_phases in invalid_sequences:
            self.write_phases(run, invalid_phases)
            with self.assertRaises(ValueError):
                cold_completion_seconds(run)

    def test_wrong_process_mode_and_changed_phase_bytes_cannot_supply_cold_time(self):
        run, phases = self.fixture()
        for field, value in (("pid", 4243), ("model", "another-model"), ("diagnostics", True)):
            invalid = copy.deepcopy(phases)
            invalid[4][field] = value
            self.write_phases(run, invalid)
            with self.assertRaises(ValueError):
                cold_completion_seconds(run)
        self.write_phases(run, phases)
        (self.root / "phases.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            cold_completion_seconds(run)

    def test_required_missing_runs_keep_the_additive_metric_unverified(self):
        result = evaluate([], .9, [("fixture", "cpu")])["fixture/cpu"]
        self.assertEqual("unfinished", result["status"])
        self.assertEqual("unverified", result["metrics"]["launch_to_first_epoch_end_seconds"]["status"])


if __name__ == "__main__":
    unittest.main()
