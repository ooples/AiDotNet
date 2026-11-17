"""Synthetic parser fixtures only; these are not benchmark/build/correctness receipts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finish_gate import cache_environment, evaluate
from measurement_contracts import CACHE_SENSITIVE_ENVIRONMENT, ROLES, SCOPES, allocation_metrics, audit_coverage, cache_coverage, checked_phases, digest, provider_values


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = {"epochs": 1, "inference_batch_sizes": [1]}
        self.artifacts = {}
        self.paths = {}
        for name in ("AiDotNet", "Tensors", "harness", "pytorch_benchmark", "pytorch_telemetry", "clr_array_profiler", "torch_runtime"):
            path = self.root / (name + ".fixture")
            path.write_text("Synthetic " + name)
            self.paths[name] = str(path)
            self.artifacts[name] = digest(path)
        self.contract = {
            "schema": "parity-measurement-contract-v1", "model": "fixture", "device": "cpu", "protocol": copy.deepcopy(self.protocol),
            "binding": {"repositories": {"fixture": "0" * 40}, "artifacts": self.artifacts.copy()},
            "allocations": {}, "caches": {},
        }
        for side in ("aidotnet", "pytorch"):
            claims = {}
            for category in ("cpu_framework_requested", "cpu_backing"):
                providers = ["torch_cpu_memory_events"] if side == "pytorch" else ["clr_arrays"]
                if side == "aidotnet" and category == "cpu_framework_requested":
                    providers.append("aidotnet_cpu_requests")
                routes = {}
                for role in ROLES[category]:
                    provider = providers[0] if role == "managed_fresh" else "aidotnet_cpu_requests"
                    status = "inactive" if role == "framework_owned_vendor_workspaces" else "measured"
                    if side == "pytorch":
                        provider = providers[0]
                        status = "measured" if role == "managed_fresh" else "excluded_lower_bound"
                    routes[role] = {"status": status, "provider": provider, **self.audit()}
                claims[category] = {
                    "scope": SCOPES[category], "window": "fixed_work_construction_training_epochs_inference_steady",
                    "bound": "upper" if side == "aidotnet" else "lower", "providers": providers,
                    "exclusions": ["vendor_internal_allocations_bypassing_framework_allocators"],
                    "unmeasured_routes": [], "routes": routes, "reviewer": "synthetic-unit-fixture",
                }
            self.contract["allocations"][side] = claims
            names = {"process_local", "vendor_process_local"}
            names |= {"cpu_autotune", "codegen_autotune", "gpu_profile"} if side == "aidotnet" else {
                "inductor", "triton", "remote_fxgraph", "remote_autotune", "external_kernel_overrides"}
            roots = {name: {"status": "inactive", **self.audit()} for name in names}
            for name in ("cpu_autotune", "codegen_autotune") if side == "aidotnet" else ("inductor", "triton"):
                roots[name]["status"] = "isolated"
            for name in ("process_local", "vendor_process_local"):
                roots[name]["status"] = "process_local"
            self.contract["caches"][side] = {"reviewer": "synthetic-unit-fixture", "unknown_roots": [], "roots": roots,
                "sensitive_environment": dict.fromkeys(CACHE_SENSITIVE_ENVIRONMENT)}
        self.commit_contract()

    @staticmethod
    def audit():
        return {"rationale": "Synthetic route coverage fixture, no production coverage claim.",
                "callsites": [{"artifact": "Tensors", "location": "synthetic:unit-fixture"}]}

    def commit_contract(self):
        path = self.root / "contract.json"
        path.write_text(json.dumps(self.contract))
        self.contract_hash = digest(path)

    def run_fixture(self, side="aidotnet", pair=0, mode="diagnostic"):
        directory = self.root / f"{side}-{mode}-{pair}"
        directory.mkdir()
        pid = (1000 if side == "aidotnet" else 2000) + pair + (100 if mode == "score" else 0)
        names = ["construction", "construction-complete", "training-prepare", "training-epoch-0",
                 "training-epoch-0-complete", "training-complete", "inference-b1-prepare", "inference-b1-warmup",
                 "inference-b1-steady", "inference-b1-complete", "complete"]
        phases = []
        for index, name in enumerate(names):
            count = (index + 1) * 2
            phases.append({"phase": name, "pid": pid, "model": "fixture", "diagnostics": mode == "diagnostic",
                "unix_ms": 100 + index,
                "clr_arrays": {"coverage": "clr_array_callbacks_v1", "abi": 1,
                    "snapshot": [1000 + count, count, count * 24, 0, 0, 0, 1]},
                "allocator_diagnostics": {"enabled": True, "cpu_storage_requested": {
                    "bytes": (index + 1) * 8, "count": index + 1, "coverage": "partial_numeric_storage_request_boundaries"}}})
        (directory / "phases.jsonl").write_text("\n".join(map(json.dumps, phases)))
        arrays = {"schema": "clr-array-profiler-v1", "pid": pid, "shutdown_complete": True, "event_mask": 0x800100,
            "snapshot": phases[-1]["clr_arrays"]["snapshot"], "sampling": False, "unmanaged_coverage": False}
        (directory / "array-allocations.json").write_text(json.dumps(arrays))
        events = [{"name": "aidotnet.phase/fixture/" + name, "ts": (index + 1) * 100, "dur": 1}
                  for index, name in enumerate(names)]
        for start in (100, 400, 900):
            events.extend({"name": "[memory]", "ts": start + 2 + index,
                           "args": {"Device Type": 0, "Bytes": 256}} for index in range(20))
        # Adaptive warmup is deliberately enormous; it must not change matched fixed-work allocation ratios.
        events.append({"name": "[memory]", "ts": 850, "args": {"Device Type": 0, "Bytes": 10000000}})
        for event in events:
            event["pid"] = pid
        (directory / "cpu-allocations.json").write_text(json.dumps({"traceEvents": events}))
        env, state = cache_environment(directory, {"regime": "cold"}, side)
        files = {"phases": "phases.jsonl", "clr_arrays": "array-allocations.json", "cpu_allocations": "cpu-allocations.json"}
        return {"path": str(directory), "pid": pid, "process_create_time": float(pid), "model": "fixture", "device": "cpu",
            "side": side, "pair": pair, "mode": mode, "return_code": 0, "protocol": copy.deepcopy(self.protocol),
            "repositories": {"fixture": "0" * 40}, "artifacts": dict(self.artifacts, measurement_contract=self.contract_hash),
            "artifact_paths": dict(self.paths, measurement_contract=str(self.root / "contract.json")),
            "diagnostic_files": {key: {"name": name, "sha256": digest(directory / name)} for key, name in files.items()},
            "cache_state": state, "environment_overrides": env,
            "cache_sensitive_environment": dict.fromkeys(CACHE_SENSITIVE_ENVIRONMENT), "machine": "synthetic", "gpu": None,
            "cache_policy": {"regime": "cold"}}

    def rewrite_raw(self, run, key, value):
        receipt = run["diagnostic_files"][key]
        path = Path(run["path"]) / receipt["name"]
        path.write_text(json.dumps(value) if key != "phases" else "\n".join(map(json.dumps, value)))
        receipt["sha256"] = digest(path)

    def test_bounds_calculate_bytes_and_counts_from_fixed_work_raw_events(self):
        ours, errors = allocation_metrics(self.run_fixture())
        self.assertEqual([], errors)
        self.assertEqual(224, ours["cpu_framework_requested_bytes"])
        self.assertEqual(12, ours["cpu_framework_requested_count"])
        self.assertEqual(192, ours["cpu_backing_bytes"])
        self.assertEqual(8, ours["cpu_backing_count"])
        theirs, errors = allocation_metrics(self.run_fixture("pytorch"))
        self.assertEqual([], errors)
        self.assertEqual(60 * 256, theirs["cpu_framework_requested_bytes"])
        self.assertEqual(60, theirs["cpu_framework_requested_count"])

    def test_missing_pool_route_or_array_as_reused_pool_coverage_is_rejected(self):
        claim = self.contract["allocations"]["aidotnet"]["cpu_framework_requested"]
        for change in (lambda: claim["routes"].pop("framework_pool_borrows"),
                       lambda: claim["routes"]["framework_pool_borrows"].update(provider="clr_arrays")):
            original = copy.deepcopy(claim)
            change()
            self.commit_contract()
            run = self.run_fixture(pair=len(list(self.root.glob("aidotnet-*"))))
            self.assertNotIn("cpu_framework_requested_bytes", allocation_metrics(run)[0])
            claim.clear()
            claim.update(original)

    def test_wrong_bound_scope_exclusions_source_or_protocol_never_pass(self):
        for index, mutation in enumerate((
            lambda c: c["allocations"]["aidotnet"]["cpu_backing"].update(bound="lower"),
            lambda c: c["allocations"]["aidotnet"]["cpu_backing"].update(scope="physical_os_heap_bytes"),
            lambda c: c["allocations"]["aidotnet"]["cpu_backing"].update(exclusions=[]),
            lambda c: c["binding"]["repositories"].update(fixture="1" * 40),
            lambda c: c["protocol"].update(epochs=2),
        )):
            original = copy.deepcopy(self.contract)
            mutation(self.contract)
            self.commit_contract()
            values, errors = allocation_metrics(self.run_fixture(pair=index))
            self.assertNotIn("cpu_backing_bytes", values)
            self.assertTrue(errors)
            self.contract = original

    def test_size_failure_missing_phase_and_tampered_raw_file_are_rejected(self):
        for index, failure in enumerate(("size", "phase", "tamper")):
            run = self.run_fixture(pair=index)
            path = Path(run["path"])
            if failure == "size":
                receipt = json.loads((path / "array-allocations.json").read_text())
                receipt["snapshot"][5] = 1
                self.rewrite_raw(run, "clr_arrays", receipt)
            elif failure == "phase":
                phases = [json.loads(line) for line in (path / "phases.jsonl").read_text().splitlines()]
                self.rewrite_raw(run, "phases", phases[:-2] + phases[-1:])
            else:
                (path / "array-allocations.json").write_text("{}")
            values, errors = allocation_metrics(run)
            self.assertNotIn("cpu_backing_count", values)
            self.assertTrue(errors)

    def test_partial_coverage_or_counter_reset_is_rejected(self):
        self.contract["allocations"]["aidotnet"]["cpu_framework_requested"]["unmeasured_routes"] = ["private-arena"]
        self.commit_contract()
        self.assertNotIn("cpu_framework_requested_count", allocation_metrics(self.run_fixture())[0])
        run = self.run_fixture(pair=1)
        phases = [json.loads(line) for line in (Path(run["path"]) / "phases.jsonl").read_text().splitlines()]
        phases[-1]["clr_arrays"]["snapshot"][1:3] = [0, 0]
        self.rewrite_raw(run, "phases", phases)
        self.assertNotIn("cpu_backing_count", allocation_metrics(run)[0])

    def test_foreign_trace_pid_cannot_supply_the_reference_lower_bound(self):
        run = self.run_fixture("pytorch")
        path = Path(run["path"]) / "cpu-allocations.json"
        trace = json.loads(path.read_text())
        for event in trace["traceEvents"]:
            event["pid"] += 1
        self.rewrite_raw(run, "cpu_allocations", trace)
        values, errors = allocation_metrics(run)
        self.assertEqual({}, values)
        self.assertTrue(any("PID" in error for error in errors))

    def test_cache_inventory_can_pass_and_unknown_or_shared_roots_cannot(self):
        run = self.run_fixture(mode="score")
        self.assertEqual("pass", cache_coverage(run)["status"])
        run["environment_overrides"]["AIDOTNET_AUTOTUNE_CACHE_PATH"] = "C:/shared"
        self.assertEqual("unverified", cache_coverage(run)["status"])
        run = self.run_fixture(mode="score", pair=1)
        run["cache_state"]["initial_files"] = {"preexisting": "hash"}
        self.assertEqual("unverified", cache_coverage(run)["status"])

    def test_remote_cache_cannot_be_declared_inactive_by_a_boolean(self):
        self.contract["caches"]["pytorch"]["sensitive_environment"]["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "1"
        self.commit_contract()
        run = self.run_fixture("pytorch", mode="score")
        run["cache_sensitive_environment"]["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "1"
        self.assertEqual("unverified", cache_coverage(run)["status"])

    def test_async_cuda_api_acquisitions_cannot_be_relabeled_physical_backing(self):
        run = self.run_fixture()
        claim = {"scope": SCOPES["gpu_backing"], "window": "fixed_work_construction_training_epochs_inference_steady",
                 "bound": "exact", "providers": ["aidotnet_cuda_backing_api"], "unmeasured_routes": [],
                 "exclusions": ["vendor_internal_allocations_bypassing_framework_allocators"],
                 "reviewer": "synthetic-unit-fixture", "routes": {
                     "backend_allocation_api": {"status": "measured", "provider": "aidotnet_cuda_backing_api", **self.audit()},
                     "framework_owned_vendor_workspaces": {"status": "inactive", **self.audit()}}}
        self.assertEqual(["aidotnet_cuda_backing_api"], audit_coverage("gpu_backing", claim, run))
        claim["scope"] = "physical_driver_pool_expansion_events"
        with self.assertRaises(ValueError):
            audit_coverage("gpu_backing", claim, run)

    def test_compiled_workspace_loans_are_requests_and_cannot_be_backing(self):
        run = self.run_fixture()
        phases = checked_phases(run)
        for index, event in enumerate(phases):
            event["allocator_diagnostics"]["gpu_compiled_workspace_loan_requested"] = {
                "bytes": (index + 1) * 64, "count": index + 1,
                "coverage": "successful_compiled_broker_loans_only"}
        values = provider_values(run, phases, "aidotnet_compiled_workspace_loans")
        self.assertEqual(256, sum(value["bytes"] for value in values.values()))
        self.assertEqual(4, sum(value["count"] for value in values.values()))
        claim = copy.deepcopy(self.contract["allocations"]["aidotnet"]["cpu_backing"])
        claim.update(scope=SCOPES["gpu_backing"], providers=["aidotnet_compiled_workspace_loans"])
        with self.assertRaises(ValueError):
            audit_coverage("gpu_backing", claim, run)

    def test_allocation_count_fails_independently_of_byte_win(self):
        runs = [self.run_fixture(side, pair, mode) for pair in range(5)
                for side in ("aidotnet", "pytorch") for mode in ("score", "diagnostic")]
        for run in runs:
            if run["side"] == "pytorch" and run["mode"] == "diagnostic":
                path = Path(run["path"]) / "cpu-allocations.json"
                trace = json.loads(path.read_text())
                trace["traceEvents"] = [event for event in trace["traceEvents"] if event["name"] != "[memory]" or event["ts"] in (102, 402, 902, 850)]
                self.rewrite_raw(run, "cpu_allocations", trace)
        with patch("finish_gate.metrics", return_value=({}, [])), patch("finish_gate.correctness_gate", return_value={"status": "pass"}):
            result = evaluate(runs, .9)["fixture/cpu"]["metrics"]
        self.assertEqual("pass", result["cpu_framework_requested_bytes"]["status"])
        self.assertEqual("fail", result["cpu_framework_requested_count"]["status"])

    def test_real_evaluator_can_pass_computable_allocation_and_cache_gates(self):
        runs = [self.run_fixture(side, pair, mode) for pair in range(5)
                for side in ("aidotnet", "pytorch") for mode in ("score", "diagnostic")]
        with patch("finish_gate.metrics", return_value=({}, [])), patch("finish_gate.correctness_gate", return_value={"status": "pass"}):
            result = evaluate(runs, .9)["fixture/cpu"]["metrics"]
        for name in ("cpu_framework_requested_bytes", "cpu_framework_requested_count", "cpu_backing_bytes", "cpu_backing_count", "first_epoch_cache_coverage"):
            self.assertEqual("pass", result[name]["status"], result[name])


if __name__ == "__main__":
    unittest.main()
