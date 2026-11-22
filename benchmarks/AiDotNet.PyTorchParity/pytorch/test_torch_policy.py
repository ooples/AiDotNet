import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch_policy


class PolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.benchmark = self.root / "fixture.py"
        self.benchmark.write_text("# Synthetic policy fixture; no model execution.\n", encoding="utf-8")
        self.receipt = self.root / "policy.json"
        self.manifest = {"torch_version": "fixture", "tf32": False, "deterministic_algorithms": True,
            "deterministic_warn_only": True, "device": "cuda", "benchmark_source_sha256": torch_policy.sha(self.benchmark)}
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.calls = []
        self.determinism = {"enabled": False, "warn_only": False}
        cuda = SimpleNamespace(is_initialized=lambda: False,
                               get_allocator_backend=lambda: self.fail("Observation initialized/probed inactive CUDA"))
        self.torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"), cuda=cuda,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True, benchmark=False, deterministic=False)),
            get_float32_matmul_precision=lambda: "highest",
            are_deterministic_algorithms_enabled=lambda: self.determinism["enabled"],
            is_deterministic_algorithms_warn_only_enabled=lambda: self.determinism["warn_only"],
            use_deterministic_algorithms=self.set_determinism)

    def set_determinism(self, value, warn_only):
        self.calls.append((value, warn_only))
        self.determinism.update(enabled=value, warn_only=warn_only)

    def invoke(self, mode, environment_path=None):
        argv = ["torch_policy.py", "--benchmark", str(self.benchmark), "--receipt", str(self.receipt),
                "--mode", mode, "--reference-manifest", str(self.manifest_path)]
        if environment_path:
            argv += ["--reference-environment", str(environment_path)]
        argv += ["--", "--device", "cuda"]
        registered = []
        # main() adds the benchmark directory to sys.path; a copy keeps that out of later tests.
        with patch("sys.argv", argv), patch("sys.path", list(sys.path)), patch.dict("sys.modules", torch=self.torch), \
             patch("torch_policy.atexit.register", side_effect=registered.append), \
             patch("torch_policy.runpy.run_path", return_value={}) as run, \
             patch("torch_policy.loaded_native_files", return_value={"status": "fixture", "files": []}):
            torch_policy.main()
            registered[0]()
        return json.loads(self.receipt.read_text()), run

    def test_observation_never_changes_defaults_or_environment_or_initializes_cuda(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "fixture-inherited"}):
            receipt, run = self.invoke("observe")
            self.assertEqual("fixture-inherited", os.environ["CUBLAS_WORKSPACE_CONFIG"])
        self.assertEqual([], self.calls)
        self.assertTrue(self.torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(self.torch.backends.cudnn.allow_tf32)
        self.assertEqual([], receipt["setters"])
        self.assertEqual("mismatch", receipt["policy_match_after"]["status"])
        self.assertTrue(receipt["atexit_complete"])
        run.assert_called_once_with(str(self.benchmark.resolve()), run_name="__main__")

    def test_strict_cuda_requires_real_reference_environment_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "startup-environment receipt"):
            self.invoke("strict-reference")
        self.assertEqual([], self.calls)

    def test_disclosed_strict_policy_records_defaults_and_explicit_reference_environment(self):
        environment = self.root / "reference-environment.json"
        environment.write_text(json.dumps({"reference_manifest_sha256": torch_policy.sha(self.manifest_path),
            "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "evidence": [{"path": str(self.benchmark), "sha256": torch_policy.sha(self.benchmark)}]}))
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "fixture-inherited"}):
            receipt, _ = self.invoke("strict-reference", environment)
            self.assertEqual(":4096:8", os.environ["CUBLAS_WORKSPACE_CONFIG"])
        self.assertEqual([(True, True)], self.calls)
        self.assertTrue(receipt["default_flags_before_public_setters"]["matmul_allow_tf32"])
        self.assertFalse(receipt["applied_flags_before_benchmark"]["matmul_allow_tf32"])
        self.assertEqual("match", receipt["policy_match_before"]["status"])
        self.assertEqual("match", receipt["policy_match_after"]["status"])
        self.assertEqual(4, len(receipt["setters"]))

    def test_changed_reference_environment_evidence_is_rejected_before_policy_setters(self):
        environment = self.root / "reference-environment.json"
        environment.write_text(json.dumps({"reference_manifest_sha256": torch_policy.sha(self.manifest_path),
            "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            "evidence": [{"path": str(self.benchmark), "sha256": "0" * 64}]}))
        with self.assertRaisesRegex(ValueError, "evidence hashes"):
            self.invoke("strict-reference", environment)
        self.assertEqual([], self.calls)

    def test_native_hashes_stream_instead_of_materializing_large_images(self):
        path = self.root / "large-image.fixture"
        expected = hashlib.sha256()
        block = b"x" * (1024 * 1024)
        with path.open("wb") as stream:
            for _ in range(3):
                stream.write(block)
                expected.update(block)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Large native image was materialized")):
            self.assertEqual(expected.hexdigest(), torch_policy.sha(path))


if __name__ == "__main__":
    unittest.main()
