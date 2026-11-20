import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from lstm_equivalence import _check_reference_compile, _initial_lstm_parameters, compare_lstm_trajectory


def write_f32(path, values):
    header, payload = {}, bytearray()
    for name, value in values.items():
        start = len(payload)
        payload.extend(value.astype("<f4").tobytes())
        header[name] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [start, len(payload)]}
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + payload)


class LstmEquivalenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        initial = {name: np.full(shape, 8, dtype=np.float32) for name, shape in (
            ("lstm.weight_ih_l0", (256, 32)), ("lstm.weight_hh_l0", (256, 64)),
            ("lstm.bias_ih_l0", (256,)), ("lstm.bias_hh_l0", (256,)), ("head.weight", (10, 64)),
            ("head.bias", (10,)))}
        initial["lstm.bias_hh_l0"].fill(0)
        # PyTorch packs the gates as i, f, g, o in 64-row blocks. Each block gets its own value, and the canonical
        # expectation is written out here, so a scorer that swaps two gate slices cannot agree with itself.
        self.gates = {"i": 1, "f": 2, "g": 3, "o": 4}
        parameters = {"head/weight": np.full((10, 64), 8, dtype=np.float32),
                      "head/bias": np.full((10,), 8, dtype=np.float32)}
        for block, (gate, value) in enumerate(self.gates.items()):
            rows = slice(block * 64, (block + 1) * 64)
            initial["lstm.weight_ih_l0"][rows] = value
            initial["lstm.weight_hh_l0"][rows] = value + 10
            initial["lstm.bias_ih_l0"][rows] = value + 20
            parameters[f"lstm/{gate}/weight_ih"] = np.full((64, 32), value, dtype=np.float32)
            parameters[f"lstm/{gate}/weight_hh"] = np.full((64, 64), value + 10, dtype=np.float32)
            parameters[f"lstm/{gate}/bias"] = np.full((64,), value + 20, dtype=np.float32)
        self.initial, self.parameters = initial, parameters
        write_f32(self.root / "initial", initial)
        self.observations = {f"inference/b{n}/logits": np.ones((n, 10), dtype=np.float32) for n in (1, 8, 16)}
        for step in range(1, 4):
            prefix = f"train/step{step}/"
            self.observations[prefix + "logits"] = np.ones((64, 10), dtype=np.float32)
            self.observations[prefix + "loss"] = np.array(.4, dtype=np.float32)
            for key, value in parameters.items():
                self.observations[prefix + "gradient/" + key] = np.ones_like(value)
                self.observations[prefix + "state/" + key] = value - step * np.float32(.001)
        write_f32(self.root / "reference", self.observations)

    def score(self):
        write_f32(self.root / "candidate", self.observations)
        return compare_lstm_trajectory(self.root / "candidate", self.root / "reference", self.root / "initial")

    def test_each_raw_gate_block_maps_to_its_own_canonical_key(self):
        canonical = _initial_lstm_parameters(self.initial)
        self.assertEqual(set(self.parameters), set(canonical))
        for key, expected in self.parameters.items():
            np.testing.assert_array_equal(expected, canonical[key], err_msg=key)

    def test_three_real_updates_and_every_inference_batch(self):
        self.assertEqual({"status": "pass", "checks": 93, "failures": {}}, self.score())

    def test_repeated_weight_and_reversed_gradient_fail_separately(self):
        self.observations["train/step3/state/lstm/f/weight_hh"] = self.observations["train/step2/state/lstm/f/weight_hh"]
        self.observations["train/step3/gradient/head/bias"] *= -1
        result = self.score()
        self.assertEqual("fail", result["status"])
        self.assertEqual({"train/step3/state/lstm/f/weight_hh", "train/step3/gradient/head/bias"}, set(result["failures"]))

    def test_missing_or_nonfinite_observations_do_not_establish_equivalence(self):
        self.observations["train/step1/loss"] = np.array(np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.score()
        del self.observations["inference/b8/logits"]
        with self.assertRaisesRegex(ValueError, "93"):
            self.score()

    def test_reference_graph_must_execute_for_each_shape_without_tf32(self):
        signatures = {key: {"graph_sha256": "a"*64, "recurrent_nodes": 1, "executions": 3} for key in
                      ("train-b64", "inference-b1", "inference-b8", "inference-b16")}
        record = {"compiler": {"backend": "inductor", "fullgraph": True, "dynamic": False, "dtype": "float32",
                  "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                  "signatures": signatures}}
        _check_reference_compile(record, 3, 1)
        signatures["inference-b8"]["executions"] = 0
        with self.assertRaisesRegex(ValueError, "inference-b8"):
            _check_reference_compile(record, 3, 1)
        record["compiler"]["cudnn_allow_tf32"] = True
        with self.assertRaisesRegex(ValueError, "TF32"):
            _check_reference_compile(record, 3, 1)
