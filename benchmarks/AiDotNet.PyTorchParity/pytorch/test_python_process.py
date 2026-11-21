import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from python_process import bootstrap_argv, resolve_python_process


class PythonProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.receipt = self.root / "process.json"
        self.launch = Mock(pid=100)
        self.launch.poll.return_value = None

    def record(self, pid=200, birth=100.1, target=None):
        record = {"schema": "benchmark-python-process-v1", "pid": pid, "process_create_time": birth}
        if target is not None:
            record.update(target=str(target), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        self.receipt.write_text(json.dumps(record), encoding="utf-8")

    @staticmethod
    def process(pid, birth, parents=()):
        value = Mock(pid=pid)
        value.create_time.return_value = birth
        value.parents.return_value = [Mock(pid=parent) for parent in parents]
        value.exe.return_value = "real-python.exe"
        return value

    def test_redirector_descendant_is_the_measured_pid_and_launcher_stays_separate(self):
        self.record()
        with patch("python_process.psutil.Process", side_effect=[self.process(100, 100), self.process(200, 100.1, [100])]):
            identity = resolve_python_process(self.launch, self.receipt)
        self.assertEqual(200, identity["pid"])
        self.assertEqual(100, identity["launcher_pid"])
        self.assertEqual(100.1, identity["process_create_time"])

    def test_direct_interpreter_pid_is_supported(self):
        self.record(pid=100, birth=100)
        with patch("python_process.psutil.Process", side_effect=[self.process(100, 100), self.process(100, 100)]):
            self.assertEqual(100, resolve_python_process(self.launch, self.receipt)["pid"])

    def test_unrelated_or_reused_pid_is_rejected_and_only_the_owned_launch_is_stopped(self):
        for actual in (self.process(200, 100.1, []), self.process(200, 101, [100])):
            self.record()
            with patch("python_process.psutil.Process", side_effect=[self.process(100, 100), actual]), patch("python_process.stop_launch") as stop:
                with self.assertRaises(ValueError):
                    resolve_python_process(self.launch, self.receipt)
                stop.assert_called_once_with(self.launch)

    def test_process_cannot_predate_its_launcher(self):
        self.record(birth=99)
        with patch("python_process.psutil.Process", side_effect=[self.process(100, 100), self.process(200, 99, [100])]), patch("python_process.stop_launch"):
            with self.assertRaisesRegex(ValueError, "predates"):
                resolve_python_process(self.launch, self.receipt)

    def test_missing_handshake_is_not_a_launcher_memory_measurement(self):
        with patch("python_process.psutil.Process", return_value=self.process(100, 100)), patch("python_process.stop_launch") as stop:
            with self.assertRaisesRegex(ValueError, "timed out"):
                resolve_python_process(self.launch, self.receipt, timeout=0)
            stop.assert_called_once_with(self.launch)

    def test_fresh_bootstrap_path_is_required(self):
        target = self.root / "target.py"
        target.write_text("pass\n", encoding="utf-8")
        self.receipt.write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fresh"):
            bootstrap_argv(["python", str(target)], __file__, self.receipt)

    def test_changed_target_bytes_cannot_supply_an_identity_receipt(self):
        target = self.root / "target.py"
        target.write_text("original", encoding="utf-8")
        self.record(target=target)
        # The recorded bytes resolve: the rejection below is for the changed bytes, not for an absent hash.
        with patch("python_process.psutil.Process", side_effect=[self.process(100, 100), self.process(200, 100.1, [100])]):
            self.assertEqual(200, resolve_python_process(self.launch, self.receipt, expected_target=target)["pid"])
        target.write_text("changed", encoding="utf-8")
        with patch("python_process.psutil.Process", return_value=self.process(100, 100)), patch("python_process.stop_launch"):
            with self.assertRaisesRegex(ValueError, "source differs"):
                resolve_python_process(self.launch, self.receipt, expected_target=target)


if __name__ == "__main__":
    unittest.main()
