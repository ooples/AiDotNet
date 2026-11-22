"""Compare complete, matched LSTM trajectories against an executed PyTorch reference."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct


def _float_tensors(path: Path):
    import numpy as np

    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("Missing safetensors header.")
    end = 8 + struct.unpack_from("<Q", raw)[0]
    if end > len(raw):
        raise ValueError("Truncated safetensors header.")
    header = json.loads(raw[8:end])
    tensors, ranges = {}, []
    for name, field in _object(header, "The safetensors header").items():
        if name == "__metadata__":
            continue
        shape = field["shape"]
        start, stop = field["data_offsets"]
        if (field["dtype"] != "F32" or not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= len(raw) - end
                or stop - start != math.prod(shape) * 4):
            raise ValueError("Invalid F32 safetensors field: " + name)
        ranges.append((start, stop))
        tensors[name] = np.frombuffer(raw, dtype="<f4", count=math.prod(shape), offset=end + start).reshape(shape)
    cursor = 0
    for start, stop in sorted(ranges):
        if start != cursor:
            raise ValueError("Safetensors payload contains a gap or overlapping fields.")
        cursor = stop
    if cursor != len(raw) - end:
        raise ValueError("Safetensors payload contains unclaimed bytes.")
    return tensors


def _initial_lstm_parameters(initial):
    import numpy as np

    shapes = {"lstm.weight_ih_l0": (256, 32), "lstm.weight_hh_l0": (256, 64),
              "lstm.bias_ih_l0": (256,), "lstm.bias_hh_l0": (256,),
              "head.weight": (10, 64), "head.bias": (10,)}
    if set(initial) != set(shapes) or any(initial[name].shape != shape or not np.isfinite(initial[name]).all()
                                          for name, shape in shapes.items()):
        raise ValueError("LSTM initial state has missing, unexpected or invalid weights.")
    if np.any(initial["lstm.bias_hh_l0"] != 0):
        raise ValueError("The frozen second LSTM bias must be zero.")
    canonical = {"head/weight": initial["head.weight"], "head/bias": initial["head.bias"]}
    for index, gate in enumerate(("i", "f", "g", "o")):
        rows = slice(index * 64, (index + 1) * 64)
        for name, source in (("weight_ih", "weight_ih"), ("weight_hh", "weight_hh"), ("bias", "bias_ih")):
            canonical[f"lstm/{gate}/{name}"] = initial[f"lstm.{source}_l0"][rows]
    return canonical


def compare_lstm_trajectory(candidate_file: Path, reference_file: Path, initial_file: Path) -> dict:
    """Recompute the 93 comparisons, including each optimizer update relative to its previous state."""
    import numpy as np

    candidate, reference = _float_tensors(candidate_file), _float_tensors(reference_file)
    initial = _initial_lstm_parameters(_float_tensors(initial_file))
    shapes = {f"inference/b{batch}/logits": (batch, 10) for batch in (1, 8, 16)}
    for step in range(1, 4):
        prefix = f"train/step{step}/"
        shapes[prefix + "logits"] = (64, 10)
        shapes[prefix + "loss"] = ()
        for name, parameter in initial.items():
            shapes[prefix + "gradient/" + name] = parameter.shape
            shapes[prefix + "state/" + name] = parameter.shape
    if set(candidate) != set(shapes) or set(reference) != set(shapes):
        raise ValueError("Expected all 93 LSTM observations on both frameworks.")
    failures = {}
    reference_activity = set()
    for name, shape in shapes.items():
        a, b = candidate[name], reference[name]
        if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("LSTM observation shape or finite-value check failed: " + name)
        a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        if "/state/" in name:
            step = int(name.split("/")[1][4:])
            parameter = name.split("/state/")[1]
            if step == 1:
                previous_a = previous_b = initial[parameter]
            else:
                previous_name = f"train/step{step - 1}/state/{parameter}"
                previous_a, previous_b = candidate[previous_name], reference[previous_name]
            a -= previous_a.ravel()
            b -= previous_b.ravel()
        if "/gradient/" in name or "/state/" in name:
            # Use the same directional and relative update criteria as the original first-step check.
            # A missing update cannot pass simply because the full weights remain close.
            aa, bb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if bb > 0:
                reference_activity.add(tuple(name.split("/")[:3]))
            if aa == 0 or bb == 0:
                if aa != bb:
                    failures[name] = {"candidate_norm": aa, "reference_norm": bb}
            else:
                relative = float(np.linalg.norm(a - b)) / bb
                cosine = float(np.dot(a, b)) / (aa * bb)
                if relative > .01 or cosine < .999:
                    failures[name] = {"relative_l2": relative, "cosine": cosine}
        else:
            error = float(np.max(np.abs(a - b)))
            if error > 1e-4:
                failures[name] = {"max_absolute_error": error}
    if any(("train", f"step{step}", kind) not in reference_activity for step in range(1, 4)
           for kind in ("gradient", "state")):
        raise ValueError("The compiled reference must produce gradients and a learned update at every step.")
    return {"status": "fail" if failures else "pass", "checks": len(shapes), "failures": failures}

def _hash(path: Path) -> str:
    from measurement_contracts import digest
    return digest(path)


def _bound_file(parent: Path, entry: dict, *, nested=False) -> Path:
    name = entry["file"]
    relative = Path(name)
    if relative.is_absolute() or (not nested and relative.name != name):
        raise ValueError("Expected a relative LSTM evidence filename.")
    target = (parent / relative).resolve()
    if not target.is_relative_to(parent.resolve()) or _hash(target) != entry["sha256"]:
        raise ValueError("LSTM evidence path or checksum differs: " + name)
    return target


def _fixture(run):
    manifest = Path(run["artifact_paths"]["lstm_fixture_manifest"])
    if _hash(manifest) != run["artifacts"]["lstm_fixture_manifest"]:
        raise ValueError("The scored LSTM fixture manifest has changed.")
    initial, inputs = manifest.parent / "initial.safetensors", manifest.parent / "inputs.safetensors"
    for key, path in (("lstm_initial_state", initial), ("lstm_shared_inputs", inputs)):
        if Path(run["artifact_paths"][key]).resolve() != path.resolve() or _hash(path) != run["artifacts"][key]:
            raise ValueError("The scored LSTM fixture asset has changed: " + key)
    spec = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if (spec.get("schema") != "lstm-classification-shared-v1" or spec.get("model") != "lstm"
            or spec.get("initial_state_sha256") != run["artifacts"]["lstm_initial_state"]
            or spec.get("inputs_sha256") != run["artifacts"]["lstm_shared_inputs"]
            or spec.get("protocol") != {"epochs": 3, "train_batches": 20, "batch_size": 64, "seed": 1234,
                                        "inference_batches": [1, 8, 16]}
            or spec.get("optimizer") != {"name": "AdamW", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
                                         "weight_decay": .01, "max_grad_norm": 1.0}):
        raise ValueError("Unexpected LSTM fixture protocol, optimizer or raw files.")
    return manifest, initial, inputs


def _descriptors(tensors):
    import hashlib

    return {name: {"dtype": "F32", "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
            for name, value in tensors.items()}


def _check_reference_compile(workload, updates, inference_iterations):
    compiler = workload["compiler"]
    if (compiler.get("backend") != "inductor" or compiler.get("fullgraph") is not True or
            compiler.get("dynamic") is not False or compiler.get("dtype") != "float32" or
            compiler.get("float32_matmul_precision") != "highest" or compiler.get("cuda_matmul_allow_tf32") is not False
            or compiler.get("cudnn_allow_tf32") is not False):
        raise ValueError("Expected executed full-graph FP32 Inductor inference and training with TF32 disabled.")
    expected = {"train-b64": updates, **{f"inference-b{b}": inference_iterations for b in (1, 8, 16)}}
    signatures = compiler["signatures"]
    if set(signatures) != set(expected):
        raise ValueError("Missing or unexpected compiled LSTM signatures.")
    for label, minimum in expected.items():
        row = signatures[label]
        graph = row["graph_sha256"]
        if (not isinstance(graph, str) or len(graph) != 64 or any(c not in "0123456789abcdef" for c in graph)
                or type(row["recurrent_nodes"]) is not int or row["recurrent_nodes"] <= 0
                or type(row["executions"]) is not int or row["executions"] < minimum):
            raise ValueError("No executed recurrent Inductor graph for " + label)


def _check_consumption(run, record, side, mode, *, pid=None):
    manifest, initial, inputs = _fixture(run)
    if (record.get("schema") != "lstm-classification-consumption-v1" or record.get("model") != "lstm"
            or record.get("framework") != side or record.get("mode") != mode or record.get("device") != run["device"]
            or type(record.get("pid")) is not int or record["pid"] <= 0 or (pid is not None and record["pid"] != pid)
            or record.get("fixture_manifest_sha256") != _hash(manifest)
            or record.get("initial_state_sha256") != _hash(initial) or record.get("inputs_sha256") != _hash(inputs)):
        raise ValueError("LSTM workload receipt differs from the measured process, fixture or device.")
    sources = ("AiDotNet", "Tensors", "AiDotNet.PyTorchParity") if side == "aidotnet" else (
        "benchmark_py", "lstm_fixture_py", "torch_runtime")
    for key in sources:
        if key not in run["artifacts"] or record["source_sha256"].get(key) != run["artifacts"][key]:
            raise ValueError("LSTM runtime or source binding differs: " + key)
    parameters = _initial_lstm_parameters(_float_tensors(initial))
    samples = _float_tensors(inputs)
    sample_shapes = {f"inference/{b}/input": (b, 32, 32) for b in (1, 8, 16)}
    for index in range(20):
        sample_shapes[f"train/{index:02}/input"] = (64, 32, 32)
        sample_shapes[f"train/{index:02}/labels"] = (64,)
    if set(samples) != set(sample_shapes) or any(samples[k].shape != shape for k, shape in sample_shapes.items()):
        raise ValueError("LSTM fixture has unexpected training or inference tensors.")
    if record.get("initial") != _descriptors(parameters) or record.get("inputs") != _descriptors(samples):
        raise ValueError("LSTM initial weights or consumed input values differ.")
    if mode == "numeric":
        order, iterations = [0, 1, 2], 1
    else:
        protocol = run["protocol"]
        if (protocol["epochs"], protocol["train_batches"], protocol["batch_size"],
                protocol["inference_batch_sizes"]) != (3, 20, 64, [1, 8, 16]):
            raise ValueError("LSTM timing requires three epochs of 20 batch-64 updates and all inference batches.")
        order, iterations = list(range(20)) * 3, protocol["inference_iterations"]
    if record.get("train_order") != order or record.get("inference_batches") != [1, 8, 16]:
        raise ValueError("Observed LSTM updates or inference batches differ from the fixed workload.")
    if side == "pytorch":
        _check_reference_compile(record, len(order), iterations)
    return initial


def _object(value, what):
    """Decoded evidence must be a JSON object; anything else is invalid evidence, not a crash."""
    if not isinstance(value, dict):
        raise ValueError(what + " is not a JSON object.")
    return value


def lstm_workload_gate(run):
    from measurement_contracts import raw_file

    try:
        path = raw_file(run, "lstm_workload")
        record = _object(json.loads(path.read_text(encoding="utf-8-sig")), "The LSTM workload receipt")
        _check_consumption(run, record, run["side"], "timing", pid=run["pid"])
        return {"status": "pass"}
    except (KeyError, TypeError, ValueError, AttributeError, OSError, ImportError, struct.error) as error:
        return {"status": "unverified", "reason": str(error)}


def lstm_correctness_gate(run):
    try:
        link = run["correctness_receipt"]
        path = Path(link["path"])
        if _hash(path) != link["sha256"]:
            raise ValueError("LSTM numerical comparison binding changed.")
        match = _object(json.loads(path.read_text(encoding="utf-8-sig")), "The LSTM comparison binding")
        if match.get("schema") != "lstm-classification-comparison-v1":
            raise ValueError("Expected a paired LSTM numerical comparison binding.")
        observations = {}
        for side in ("aidotnet", "pytorch"):
            side_path = _bound_file(path.parent, match[side], nested=True)
            proof = _object(json.loads(side_path.read_text(encoding="utf-8-sig")), "The LSTM numerical receipt")
            workload = _object(json.loads(_bound_file(side_path.parent, proof["workload"]).read_text(encoding="utf-8-sig")),
                               "The LSTM workload receipt")
            initial = _check_consumption(run, workload, side, "numeric")
            # One numerical process writes both receipts. Observations from one run paired with the workload
            # receipt of another would pass every other metadata check. The separate timing run has its own PID.
            if type(proof.get("pid")) is not int or proof["pid"] != workload.get("pid"):
                raise ValueError("The LSTM numerical receipt and its workload receipt name different processes.")
            if (proof.get("schema") != "lstm-classification-numerical-v1" or proof.get("model") != "lstm"
                    or proof.get("completed_updates") != 3 or proof.get("gradient_definition") != "post_global_norm_clip_1"
                    or any(proof.get(key) != workload[key] for key in ("framework", "device", "fixture_manifest_sha256",
                       "initial_state_sha256", "inputs_sha256", "source_sha256"))):
                raise ValueError("LSTM numerical receipt does not describe the matched optimizer updates.")
            observations[side] = _bound_file(side_path.parent, proof["observations"])
        return compare_lstm_trajectory(observations["aidotnet"], observations["pytorch"], initial)
    except (KeyError, TypeError, ValueError, AttributeError, OSError, ImportError, struct.error) as error:
        return {"status": "unverified", "reason": str(error)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bind two completed numerical runs for the LSTM scorer.")
    parser.add_argument("--aidotnet", type=Path, required=True, help="C# numeric-receipt.json")
    parser.add_argument("--pytorch", type=Path, required=True, help="PyTorch numeric-receipt.json")
    parser.add_argument("--output", type=Path, required=True, help="Comparison file in a common parent directory")
    args = parser.parse_args()
    match = {"schema": "lstm-classification-comparison-v1"}
    for side in ("aidotnet", "pytorch"):
        path = getattr(args, side).resolve()
        match[side] = {"file": path.relative_to(args.output.resolve().parent).as_posix(), "sha256": _hash(path)}
    args.output.write_text(json.dumps(match, indent=2) + "\n", encoding="utf-8")
