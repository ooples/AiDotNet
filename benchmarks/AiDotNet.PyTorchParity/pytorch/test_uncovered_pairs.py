"""run_matrix.uncovered_pairs must reject a selection that leaves any model/device row without a PyTorch side."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_matrix  # noqa: E402


class UncoveredPairsTests(unittest.TestCase):
    def test_model_limited_configuration_does_not_cover_other_models(self):
        # The RNN arm runs on the CPU but only for GRU, so MLP would run with AiDotNet alone.
        self.assertEqual(run_matrix.uncovered_pairs(["cpu"], ["mlp"], ["torch-compile-fused-rnn"]), ["mlp on cpu"])

    def test_model_limited_configuration_covers_its_models(self):
        self.assertEqual(run_matrix.uncovered_pairs(["cpu"], ["gru"], ["torch-compile-fused-rnn"]), [])

    def test_device_limited_configuration_does_not_cover_other_devices(self):
        self.assertEqual(run_matrix.uncovered_pairs(["cpu", "cuda"], ["mlp"], ["torch-cudagraphs-fused"]), ["mlp on cpu"])

    def test_every_pair_is_reported(self):
        self.assertEqual(
            run_matrix.uncovered_pairs(["cpu", "cuda"], ["mlp", "gru"], ["torch-compile-fused-rnn"]),
            ["mlp on cpu", "mlp on cuda", "gru on cuda"])

    def test_default_configurations_cover_every_model_on_both_devices(self):
        self.assertEqual(run_matrix.uncovered_pairs(["cpu", "cuda"], list(run_matrix.MODELS), list(run_matrix.ARMS)), [])


if __name__ == "__main__":
    unittest.main()
