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


def torch_report(arm, model, compiled_graphs, drop_compile_rnn=False):
    protocol = dict(run_matrix.ARMS[arm]["protocol"])
    protocol.setdefault("cuda_graphs", False)
    if drop_compile_rnn:
        protocol.pop("compile_rnn", None)
    protocol["compiled_graphs"] = {model: compiled_graphs}
    return {
        "framework": "PyTorch",
        "results": [{"model": model}],
        "device": "cpu",
        "torch_num_threads": run_matrix.THREADS,
        "cuda_available": False,
        "protocol": protocol,
    }


class CheckProtocolPyTorchTests(unittest.TestCase):
    def test_report_without_compile_rnn_counts_as_false(self):
        # Reports written before --compile-rnn existed carry no compile_rnn key.
        run_matrix.check_protocol(torch_report("torch-compile", "mlp", 3, drop_compile_rnn=True), "torch-compile", "mlp", "cpu")

    def test_missing_compile_rnn_does_not_satisfy_an_rnn_arm(self):
        with self.assertRaisesRegex(ValueError, "PyTorch ran"):
            run_matrix.check_protocol(torch_report("torch-compile-fused-rnn", "gru", 2, drop_compile_rnn=True),
                                      "torch-compile-fused-rnn", "gru", "cpu")

    def test_compiled_arm_that_compiled_no_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "compiled no graph"):
            run_matrix.check_protocol(torch_report("torch-compile", "mlp", 0), "torch-compile", "mlp", "cpu")

    def test_rnn_not_told_to_trace_may_compile_no_graph(self):
        run_matrix.check_protocol(torch_report("torch-compile", "gru", 0), "torch-compile", "gru", "cpu")

    def test_rnn_told_to_trace_must_compile_a_graph(self):
        with self.assertRaisesRegex(ValueError, "compiled no graph"):
            run_matrix.check_protocol(torch_report("torch-compile-fused-rnn", "gru", 0), "torch-compile-fused-rnn", "gru", "cpu")


if __name__ == "__main__":
    unittest.main()
