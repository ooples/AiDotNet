import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finish_gate import reference_recipe_gate


class ReferencePolicyGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = {}
        self.hashes = {}
        for name in ("pytorch_benchmark", "torch_policy_wrapper"):
            path = self.root / (name + ".fixture")
            path.write_text("Synthetic policy-gate fixture: " + name, encoding="utf-8")
            self.files[name] = str(path)
            self.hashes[name] = self.sha(path)
        self.manifest = {"model": "fixture", "device": "cpu", "torch_version": "fixture",
            "tf32": False, "deterministic_algorithms": True, "deterministic_warn_only": False,
            "benchmark_source_sha256": self.hashes["pytorch_benchmark"]}
        self.run = {"model": "fixture", "device": "cpu", "side": "pytorch", "mode": "score",
            "pid": 4242, "return_code": 0, "path": str(self.root), "artifact_paths": self.files,
            "artifacts": self.hashes, "diagnostic_files": {}}
        self.write_manifest()
        flags = {"torch_version": "fixture", "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "deterministic_algorithms": True, "deterministic_warn_only": False,
            "cuda_initialized_before_observation": False, "cuda_initialized_after_observation": False}
        self.policy = {"schema": "torch-policy-native-receipt-v1", "mode": "strict-reference", "pid": 4242,
            "benchmark_exit_code": 0, "atexit_complete": True,
            "benchmark_sha256": self.hashes["pytorch_benchmark"], "wrapper_sha256": self.hashes["torch_policy_wrapper"],
            "reference_manifest_sha256": self.hashes["hifigan_reference_manifest"],
            "effective_environment": {}, "environment_after_benchmark": {}, "environment_at_exit": {},
            "native": {"status": "observed_loaded_images", "files": [{"path": "fixture-native-image.dll", "bytes": 123, "sha256": "1" * 64}]}}
        for point in ("applied_flags_before_benchmark", "flags_after_benchmark", "flags_at_exit"):
            self.policy[point] = copy.deepcopy(flags)
        self.write_policy()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_manifest(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.files["hifigan_reference_manifest"] = str(path)
        self.hashes["hifigan_reference_manifest"] = self.sha(path)

    def write_policy(self):
        path = self.root / "torch-policy.json"
        path.write_text(json.dumps(self.policy), encoding="utf-8")
        self.run["diagnostic_files"]["torch_policy"] = {"name": path.name, "sha256": self.sha(path)}

    def test_exact_actual_policy_is_required_in_addition_to_source_recipe_hash(self):
        self.assertEqual("pass", reference_recipe_gate(self.run)["status"])

    def test_observation_or_different_defaults_are_not_a_formal_reference_pass(self):
        self.policy["mode"] = "observe"
        self.write_policy()
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])
        self.policy["mode"] = "strict-reference"
        self.policy["applied_flags_before_benchmark"]["cudnn_allow_tf32"] = True
        self.write_policy()
        self.assertEqual("fail", reference_recipe_gate(self.run)["status"])

    def test_missing_or_changed_raw_receipt_cannot_pass(self):
        (self.root / "torch-policy.json").write_text("changed", encoding="utf-8")
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])
        (self.root / "torch-policy.json").unlink()
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])

    def test_wrong_pid_wrapper_or_manifest_hash_is_rejected_with_updated_raw_hash(self):
        original = copy.deepcopy(self.policy)
        for field, value in (("pid", 4243), ("wrapper_sha256", "0" * 64), ("reference_manifest_sha256", "0" * 64)):
            self.policy = dict(copy.deepcopy(original), **{field: value})
            self.write_policy()
            self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])

    def test_changed_post_flags_fail_even_if_receipt_claims_pass(self):
        self.policy["pass"] = True
        self.policy["policy_match_after"] = {"status": "match"}
        self.policy["flags_after_benchmark"]["deterministic_algorithms"] = False
        self.write_policy()
        self.assertEqual("fail", reference_recipe_gate(self.run)["status"])

    def test_override_environment_and_native_observation_must_remain_complete(self):
        original = copy.deepcopy(self.policy)
        self.policy["environment_at_exit"] = {"NVIDIA_TF32_OVERRIDE": "1"}
        self.write_policy()
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])
        self.policy = copy.deepcopy(original)
        self.policy["native"]["files"] = []
        self.write_policy()
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])

    def test_legacy_warn_only_needs_source_attestation_not_an_implicit_default(self):
        del self.manifest["deterministic_warn_only"]
        self.write_manifest()
        self.policy["reference_manifest_sha256"] = self.hashes["hifigan_reference_manifest"]
        self.write_policy()
        self.assertEqual("unverified", reference_recipe_gate(self.run)["status"])
        path = self.root / "legacy-policy.json"
        path.write_text(json.dumps({"reference_manifest_sha256": self.hashes["hifigan_reference_manifest"],
            "policy": {"deterministic_warn_only": False}, "environment": {},
            "evidence": [{"path": self.files["torch_policy_wrapper"], "sha256": self.hashes["torch_policy_wrapper"]}]}))
        self.files["reference_environment"] = str(path)
        self.hashes["reference_environment"] = self.sha(path)
        self.policy["reference_environment"] = {"path": str(path), "sha256": self.sha(path)}
        self.write_policy()
        self.assertEqual("pass", reference_recipe_gate(self.run)["status"])


if __name__ == "__main__":
    unittest.main()
