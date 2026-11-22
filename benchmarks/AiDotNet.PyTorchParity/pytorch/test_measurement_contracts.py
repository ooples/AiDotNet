"""Synthetic parser fixtures only; these are not benchmark/build/correctness receipts."""
from contextlib import closing
import sqlite3
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

    def cuda_trace_fixture(self, pair=0):
        self.protocol["epochs"] = 2
        self.contract.update(device="cuda", protocol=copy.deepcopy(self.protocol))
        self.commit_contract()
        run = self.run_fixture(pair=pair)
        directory = Path(run["path"])
        run.update(device="cuda", gpu_mapping={"uuid": "GPU-abcd", "logical_device": 0},
                   profiled_argv=["dotnet", "fixture.dll", "--output", str(directory / "report.json")])
        names = [p["phase"] for p in map(json.loads, (directory / "phases.jsonl").read_text().splitlines())]
        at = names.index("training-complete")
        names[at:at] = ["training-epoch-1", "training-epoch-1-complete"]
        self.rewrite_raw(run, "phases", [{"phase": name, "pid": run["pid"], "model": run["model"],
                                         "diagnostics": True, "unix_ms": 1000 + index * 10}
                                        for index, name in enumerate(names)])
        path = directory / "native-memory.sqlite"
        with closing(sqlite3.connect(path)) as db:
            db.executescript("""
                CREATE TABLE META_DATA_EXPORT(name,value);
                CREATE TABLE META_DATA_CAPTURE(name,value);
                CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId,cudaId,pid,uuid);
                CREATE TABLE ANALYSIS_DETAILS(globalVid,duration);
                CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs);
                CREATE TABLE DIAGNOSTIC_EVENT(globalPid,severity,text);
                CREATE TABLE StringIds(id,value);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start,end,nameId,eventClass);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid,deviceId,start,end);
                INSERT INTO META_DATA_EXPORT VALUES ('EXPORT_PRODUCT_NAME','NVIDIA Nsight Systems'),('EXPORT_SCHEMA_VERSION','3.29.0');
                INSERT INTO META_DATA_CAPTURE VALUES ('CAPTURE_EVENT_TYPE','Cuda'),('CUDA_TRACE_SCOPE','ProcessTree'),
                    ('CUDA_SKIP_SOME_API_CALLS','false'),('COLLECT_GPU_MEMORY_USAGE','true'),('DISABLE_TIME_SYNC','false'),
                    ('PROCESS_0:COMMAND','dotnet');
                INSERT INTO ANALYSIS_DETAILS VALUES (281474976710656, 300000000);
                INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (999000000);
                INSERT INTO StringIds VALUES (1,'cudaMalloc_v3020'),(2,'cuMemAllocAsync_ptsz'),(3,'cuMemFree_v2'),
                    (4,'cudaFreeAsync_v11020'),(5,'cudaFreeHost_v3020'),(6,'cuLaunchKernel');
            """)
            db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0,0,?,'abcd')", (run["pid"],))
            db.executemany("INSERT INTO META_DATA_CAPTURE VALUES (?,?)", [
                (f"PROCESS_0:ARGUMENT_{i}", arg) for i, arg in enumerate(run["profiled_argv"][1:])])
            process = (1 << 48) | (run["pid"] << 24)
            thread = process | 2
            tick = lambda name: 1_000_000 + names.index(name) * 10_000_000
            train, infer = tick("training-epoch-1"), tick("inference-b1-steady")
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,0,?,?)", [
                (process, mark + 4_000_000, mark + 5_000_000) for mark in (train, infer)])
            events = [(thread, tick("training-epoch-0") + 2_000_000, 20, 1, 0),
                      (thread, tick("inference-b1-warmup") + 2_000_000, 20, 2, 1),
                      (thread, train + 2_000_000, 200, 1, 0),   # runtime wrapper
                      (thread, train + 2_000_020, 20, 2, 1),    # actual driver acquisition
                      (thread, train + 2_000_080, 20, 2, 1),    # second driver acquisition
                      (thread, train + 3_000_000, 20, 1, 0),    # standalone runtime allocation
                      (thread, train + 4_000_000, 20, 3, 1),
                      (thread, train + 3_500_000, 20, 6, 1),
                      (thread, infer + 2_000_000, 20, 4, 0),
                      (thread, infer + 3_000_000, 20, 5, 0),    # host allocation is out of scope
                      (thread, infer + 3_500_000, 20, 6, 1),
                      (thread + (1 << 24), infer + 3_000_000, 20, 2, 1)]  # another PID
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)", [
                (tid, start, start + duration, api, domain) for tid, start, duration, api, domain in events])
            db.commit()
        run["diagnostic_files"]["cuda_nsys_sqlite"] = {"name": path.name, "sha256": digest(path)}
        return run, path

    def test_cuda_native_warmed_windows_bind_zero_and_count_nested_driver_calls_once(self):
        run, _ = self.cuda_trace_fixture()
        values, errors = allocation_metrics(run)
        self.assertFalse([e for e in errors if e.startswith("cuda_native")], errors)
        self.assertEqual({k: v for k, v in values.items() if k.startswith("cuda_native")}, {
            "cuda_native_training_epoch_1_alloc_calls": 3, "cuda_native_training_epoch_1_free_calls": 1,
            "cuda_native_inference_b1_alloc_calls": 0, "cuda_native_inference_b1_free_calls": 1})

    def test_cuda_native_verifies_true_warmed_zero_with_active_cuda_work(self):
        run, path = self.cuda_trace_fixture()
        with closing(sqlite3.connect(path)) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= 51000000 AND nameId != 6")
            db.commit()
        run["diagnostic_files"]["cuda_nsys_sqlite"]["sha256"] = digest(path)
        values, errors = allocation_metrics(run)
        self.assertFalse([e for e in errors if e.startswith("cuda_native")], errors)
        native = [v for k, v in values.items() if k.startswith("cuda_native")]
        self.assertEqual([0, 0, 0, 0], native)

    def test_cuda_native_drops_incomplete_foreign_or_ambiguous_evidence(self):
        for pair, statement in enumerate([
            "UPDATE META_DATA_CAPTURE SET value='true' WHERE name='CUDA_SKIP_SOME_API_CALLS'",
            "UPDATE TARGET_INFO_CUDA_DEVICE SET pid=pid+1",
            "UPDATE TARGET_INFO_CUDA_DEVICE SET uuid='another-device'",
            "UPDATE META_DATA_CAPTURE SET value='another/report.json' WHERE name='PROCESS_0:ARGUMENT_2'",
            "UPDATE TARGET_INFO_SESSION_START_TIME SET utcEpochNs=0",
            "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>100000000",
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE eventClass IN (0,1)",
            "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=51000100, end=51000110 WHERE nameId=3",
            "INSERT INTO DIAGNOSTIC_EVENT VALUES (0,2,'CUDA events dropped')",
            "UPDATE StringIds SET value='cuMemAllocUnsupported_v2' WHERE id=2",
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>100000000",
            "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=start+300 WHERE nameId=2 AND start BETWEEN 53000010 AND 53000060",
            "CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (id)",
        ]):
            with self.subTest(statement=statement):
                run, path = self.cuda_trace_fixture(pair)
                with closing(sqlite3.connect(path)) as db:
                    db.execute(statement)
                    db.commit()
                run["diagnostic_files"]["cuda_nsys_sqlite"]["sha256"] = digest(path)
                values, errors = allocation_metrics(run)
                self.assertTrue(any(e.startswith("cuda_native") for e in errors), errors)
                self.assertFalse(any(k.startswith("cuda_native") for k in values))

    def test_cuda_native_trace_hash_and_missing_trace_fail_closed(self):
        run, path = self.cuda_trace_fixture()
        with path.open("ab") as trace:
            trace.write(b"changed after run receipt")
        for receipt in ("tampered", "missing"):
            with self.subTest(receipt=receipt):
                if receipt == "missing":
                    del run["diagnostic_files"]["cuda_nsys_sqlite"]
                values, errors = allocation_metrics(run)
                self.assertTrue(any(e.startswith("cuda_native") for e in errors), errors)
                self.assertFalse(any(k.startswith("cuda_native") for k in values))

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
        for pair, variable in enumerate(("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE", "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE", "TRITON_KERNEL_OVERRIDE")):
            with self.subTest(variable=variable):
                self.contract["caches"]["pytorch"]["sensitive_environment"] = dict.fromkeys(CACHE_SENSITIVE_ENVIRONMENT)
                self.contract["caches"]["pytorch"]["sensitive_environment"][variable] = "1"
                self.commit_contract()
                run = self.run_fixture("pytorch", pair=pair, mode="score")
                run["cache_sensitive_environment"][variable] = "1"
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

    def test_torch_cpu_provider_counts_only_positive_cpu_events_inside_each_scored_window(self):
        run = self.run_fixture(side="pytorch")
        path = Path(run["path"]) / "cpu-allocations.json"
        trace = json.loads(path.read_text())
        pid = run["pid"]
        # Inside the construction window: a parent operator carrying its children's bytes, a free and a GPU event.
        trace["traceEvents"] += [
            {"name": "aten::add", "ts": 150, "pid": pid, "args": {"cpu_memory_usage": 4096}},
            {"name": "[memory]", "ts": 151, "pid": pid, "args": {"Device Type": 0, "Bytes": -4096}},
            {"name": "[memory]", "ts": 152, "pid": pid, "args": {"Device Type": 1, "Bytes": 2048}},
            # Positive CPU events outside every scored window: between windows and in training preparation.
            {"name": "[memory]", "ts": 250, "pid": pid, "args": {"Device Type": 0, "Bytes": 512}},
            {"name": "[memory]", "ts": 350, "pid": pid, "args": {"Device Type": 0, "Bytes": 512}},
        ]
        self.rewrite_raw(run, "cpu_allocations", trace)
        values = provider_values(run, checked_phases(run), "torch_cpu_memory_events")
        self.assertEqual({"construction", "training_epoch_0", "inference_b1"}, set(values))
        for window, value in values.items():
            self.assertEqual({"bytes": 20 * 256, "count": 20}, value, window)

    def test_missing_counter_at_a_scored_boundary_is_not_zero_without_uninitialized_evidence(self):
        run = self.run_fixture()
        phases = checked_phases(run)
        field, coverage = "gpu_compiled_workspace_loan_requested", "successful_compiled_broker_loans_only"
        for index, event in enumerate(phases):
            # Absent through construction and epoch 0, first present at the inference markers.
            if index >= 6:
                event["allocator_diagnostics"][field] = {"bytes": (index + 1) * 64, "count": index + 1, "coverage": coverage}
        with self.assertRaisesRegex(ValueError, "no counter at the scored boundary construction-complete"):
            provider_values(run, phases, "aidotnet_compiled_workspace_loans")

    def test_torch_cuda_startup_allowance_requires_recorded_uninitialized_state(self):
        run = self.run_fixture(side="pytorch")
        phases = checked_phases(run)
        source = "pytorch_native_caching_allocator_requests"
        for index, event in enumerate(phases):
            event["cuda_initialized"] = index >= 3
            if index >= 3:
                event["allocators"] = {"gpu_framework_requested": {"bytes": index * 10, "count": index, "source": source}}
        values = provider_values(run, phases, "torch_cuda_requests")
        self.assertEqual({"bytes": 0, "count": 0}, values["construction"])
        self.assertEqual({"bytes": 10, "count": 1}, values["training_epoch_0"])
        # The same series without the recorded state is unknown, not zero.
        for event in phases[:3]:
            del event["cuda_initialized"]
        with self.assertRaisesRegex(ValueError, "no counter at the scored boundary"):
            provider_values(run, phases, "torch_cuda_requests")

    def test_compiled_workspace_loans_are_requests_and_cannot_be_backing(self):
        run = self.run_fixture()
        phases = checked_phases(run)
        for provider, field, coverage in (
            ("aidotnet_compiled_workspace_loans", "gpu_compiled_workspace_loan_requested", "successful_compiled_broker_loans_only"),
            ("aidotnet_step_scratch_reuse", "gpu_step_scratch_reuse_requested", "successful_step_scratch_free_loans_only"),
        ):
            for index, event in enumerate(phases):
                event["allocator_diagnostics"][field] = {"bytes": (index + 1) * 64, "count": index + 1, "coverage": coverage}
            values = provider_values(run, phases, provider)
            self.assertEqual(256, sum(value["bytes"] for value in values.values()))
            self.assertEqual(4, sum(value["count"] for value in values.values()))
            claim = copy.deepcopy(self.contract["allocations"]["aidotnet"]["cpu_backing"])
            claim.update(scope=SCOPES["gpu_backing"], providers=[provider])
            with self.assertRaises(ValueError):
                audit_coverage("gpu_backing", claim, run)

    def test_pool_route_requires_all_of_its_distinct_counter_providers(self):
        run = self.run_fixture()
        providers = ["aidotnet_gpu_requests", "aidotnet_compiled_workspace_loans", "aidotnet_step_scratch_reuse"]
        claim = {"scope": SCOPES["gpu_framework_requested"], "window": "fixed_work_construction_training_epochs_inference_steady",
            "bound": "upper", "providers": providers, "unmeasured_routes": [], "reviewer": "synthetic-unit-fixture",
            "exclusions": ["vendor_internal_allocations_bypassing_framework_allocators"], "routes": {
                "backend_buffer_api": {"status": "measured", "provider": providers[0], **self.audit()},
                "framework_pool_borrows": {"status": "measured", "providers": providers[1:], **self.audit()},
                "framework_owned_vendor_workspaces": {"status": "inactive", **self.audit()}}}
        self.assertEqual(providers, audit_coverage("gpu_framework_requested", claim, run))
        claim["routes"]["framework_pool_borrows"]["providers"].pop()
        with self.assertRaises(ValueError):
            audit_coverage("gpu_framework_requested", claim, run)

    def test_cudnn_workspace_acquisitions_can_close_the_owned_vendor_boundary(self):
        run = self.run_fixture()
        phases = checked_phases(run)
        for index, event in enumerate(phases):
            event["allocator_diagnostics"]["gpu_cudnn_workspace_acquired"] = {
                "bytes": (index + 1) * 64, "count": index + 1,
                "coverage": "successful_cudnn_conv_workspace_growth_only"}
        values = provider_values(run, phases, "aidotnet_cudnn_workspace")
        self.assertEqual(256, sum(value["bytes"] for value in values.values()))
        claim = {"scope": SCOPES["gpu_backing"], "window": "fixed_work_construction_training_epochs_inference_steady",
            "bound": "exact", "providers": ["aidotnet_cuda_backing_api", "aidotnet_cudnn_workspace"],
            "unmeasured_routes": [], "reviewer": "synthetic-unit-fixture",
            "exclusions": ["vendor_internal_allocations_bypassing_framework_allocators"], "routes": {
                "backend_allocation_api": {"status": "measured", "provider": "aidotnet_cuda_backing_api", **self.audit()},
                "framework_owned_vendor_workspaces": {"status": "measured", "provider": "aidotnet_cudnn_workspace", **self.audit()}}}
        self.assertEqual(claim["providers"], audit_coverage("gpu_backing", claim, run))

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
