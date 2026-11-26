"""run_matrix.check_protocol must accept archived reports that predate the precision option and reject other precisions."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_matrix  # noqa: E402


def aidotnet_report(protocol=None):
    report = {
        "Framework": "AiDotNet",
        "Results": [{"Model": "mlp"}],
        "Environment": {"CpuThreads": run_matrix.THREADS, "EngineDevice": "CpuEngine", "EngineName": "CpuEngine"},
    }
    if protocol is not None:
        report["Protocol"] = protocol
    return report


class CheckProtocolPrecisionTests(unittest.TestCase):
    def test_report_without_protocol_block_counts_as_fp32(self):
        # Every run published before --precision existed has no Protocol block; --summarize must still accept it.
        run_matrix.check_protocol(aidotnet_report(), "aidotnet", "mlp", "cpu")

    def test_report_recording_fp32_is_accepted(self):
        run_matrix.check_protocol(aidotnet_report({"Precision": "fp32"}), "aidotnet", "mlp", "cpu")

    def test_report_at_another_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "precision 'bf16'"):
            run_matrix.check_protocol(aidotnet_report({"Precision": "bf16"}), "aidotnet", "mlp", "cpu")


if __name__ == "__main__":
    unittest.main()
