import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmark import make_model
from hifigan_fixture import TensorFile, descriptor, file_hash
from lstm_fixture import (ExecutedInductor, LstmFixture, PROTOCOL, canonical, generate,
                          input_shapes, strict_lstm_settings)


class LstmFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "fixture"
        generate(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self):
        args = SimpleNamespace(epochs=3, train_batches=20, batch_size=64, seed=1234, lstm_numerics=None,
                               output=Path(self.temp.name) / "unused", lstm_workload_receipt=None)
        model, _ = make_model("lstm")
        return LstmFixture(self.root, args, torch.device("cpu"), [1, 8, 16], model)

    def test_export_matches_normal_initialization_and_contains_changing_batches(self):
        m = json.loads((self.root / "manifest.json").read_text())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model, _ = make_model("lstm")
            self.assertEqual(m["initial"], {k: descriptor(v) for k, v in canonical(model.state_dict()).items()})
        self.assertEqual(14, len(m["initial"]))
        self.assertEqual(set(input_shapes()), set(m["inputs"]))
        self.assertEqual(20, len({m["inputs"][f"train/{i:02d}/input"]["sha256"] for i in range(20)}))
        self.assertEqual(20, len({m["inputs"][f"train/{i:02d}/labels"]["sha256"] for i in range(20)}))
        with TensorFile(self.root / "initial.safetensors") as values:
            self.assertEqual(6, len(values.names))
            self.assertEqual(0, values.tensor("lstm.bias_hh_l0").count_nonzero().item())

    def test_repeatable_export(self):
        second = Path(self.temp.name) / "second"
        generate(second)
        for file in ("manifest.json", "initial.safetensors", "inputs.safetensors"):
            self.assertEqual(file_hash(self.root / file), file_hash(second / file))
        with self.assertRaises(FileExistsError):
            generate(second)

    def test_consumption_rejects_wrong_order_swapped_labels_and_partial_run(self):
        fixture = self.load()
        first, second = fixture.pool[:2]
        with self.assertRaisesRegex(ValueError, "different batch"):
            fixture.trained(*second)
        with self.assertRaisesRegex(ValueError, "different batch"):
            fixture.trained(first[0], second[1])
        fixture.trained(*first)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            fixture.complete()

    def test_consumption_rechecks_input_values_at_completion(self):
        fixture = self.load()
        for _ in range(3):
            for pair in fixture.pool: fixture.trained(*pair)
        for b in PROTOCOL["inference_batches"]: fixture.inference(b)
        fixture.pool[7][0][1, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "mutated"):
            fixture.complete()

    def test_recurrent_compiler_rejects_nonrecurrent_graph_and_missing_signatures(self):
        compiler = ExecutedInductor()
        graph = torch.fx.symbolic_trace(torch.nn.Linear(32, 64))
        with self.assertRaisesRegex(RuntimeError, "capture the LSTM"):
            compiler(graph, [])
        with self.assertRaisesRegex(RuntimeError, "Missing executed"):
            compiler.evidence()

    def test_receipt_counts_only_successful_invocations_bound_to_real_batch_size(self):
        # Isolate the observer here. The independent numerical process runs the real Inductor backend.
        graph = SimpleNamespace(code="torch.lstm(x)", graph=SimpleNamespace(nodes=[
            SimpleNamespace(op="call_function", target=torch.lstm)]))
        compiler = ExecutedInductor()
        with patch("torch._dynamo.backends.registry.lookup_backend", return_value=lambda g, args: (lambda x: x)):
            guarded = [compiler(graph, []) for _ in range(4)]
        self.assertEqual({}, compiler.signatures)
        with strict_lstm_settings():
            guarded[0](torch.zeros(64, 32, 32))
            with torch.inference_mode():
                for run, b in zip(guarded[1:], (1, 8, 16)):
                    run(torch.zeros(b, 32, 32))
            receipt = compiler.evidence()
            self.assertEqual(4, len(receipt["signatures"]))
            self.assertTrue(all(row["recurrent_nodes"] == row["executions"] == 1 for row in receipt["signatures"].values()))
            self.assertEqual("highest", receipt["float32_matmul_precision"])
            self.assertFalse(receipt["cuda_matmul_allow_tf32"])
            self.assertFalse(receipt["cudnn_allow_tf32"])


if __name__ == "__main__":
    unittest.main()

