"""Shared FP32 state, workloads and Inductor execution receipts for the existing LSTM row."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch

from hifigan_fixture import TensorFile, descriptor, file_hash

PROTOCOL = {"epochs": 3, "train_batches": 20, "batch_size": 64, "seed": 1234,
            "inference_batches": [1, 8, 16]}
GRADIENT = "post_global_norm_clip_1"
OPTIMIZER = {"name": "AdamW", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
             "weight_decay": 0.01, "max_grad_norm": 1.0}


def canonical(weights):
    """PyTorch packs input, forget, candidate, output in contiguous row groups."""
    result = {}
    for target, source in (("weight_ih", "weight_ih_l0"), ("weight_hh", "weight_hh_l0"), ("bias", "bias_ih_l0")):
        for gate, values in zip("ifgo", weights["lstm." + source].chunk(4)):
            result[f"lstm/{gate}/{target}"] = values
    return {**result, "head/weight": weights["head.weight"], "head/bias": weights["head.bias"]}


def input_shapes():
    return {**{f"train/{i:02d}/{name}": shape for i in range(20)
               for name, shape in (("input", [64, 32, 32]), ("labels", [64]))},
            **{f"inference/{b}/input": [b, 32, 32] for b in (1, 8, 16)}}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)


def source_hashes():
    directory = Path(__file__).resolve().parent
    return {"benchmark_py": file_hash(directory / "benchmark.py"), "lstm_fixture_py": file_hash(Path(__file__)),
            "torch_runtime": file_hash(Path(torch._C.__file__))}


def generate(root):
    from benchmark import make_model
    from export_reference import write_safetensors
    root.mkdir(parents=True, exist_ok=True)
    if any((root / f).exists() for f in ("manifest.json", "inputs.safetensors", "initial.safetensors")):
        raise FileExistsError(root)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROTOCOL["seed"])
        model, shape = make_model("lstm")
        weights = model.state_dict()
        inputs = {}
        for i in range(20):
            inputs[f"train/{i:02d}/input"] = torch.randn(64, *shape)
            inputs[f"train/{i:02d}/labels"] = torch.randint(0, 10, (64,)).float()
        for batch in PROTOCOL["inference_batches"]:
            inputs[f"inference/{batch}/input"] = torch.randn(batch, *shape)
        write_safetensors(root / "initial.safetensors", weights)
        write_safetensors(root / "inputs.safetensors", inputs)
    save_json(root / "manifest.json", {
        "schema": "lstm-classification-shared-v1", "model": "lstm", "protocol": PROTOCOL, "optimizer": OPTIMIZER,
        "initial_state_sha256": file_hash(root / "initial.safetensors"), "inputs_sha256": file_hash(root / "inputs.safetensors"),
        "initial": {k: descriptor(v) for k, v in canonical(weights).items()},
        "inputs": {k: descriptor(v) for k, v in inputs.items()}})


@contextmanager
def strict_lstm_settings():
    old_matmul = torch.get_float32_matmul_precision()
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        with torch._dynamo.config.patch(allow_rnn=True):
            yield
    finally:
        torch.set_float32_matmul_precision(old_matmul)
        torch.backends.cudnn.allow_tf32 = old_cudnn


class ExecutedInductor:
    def __init__(self):
        self.signatures = {}

    def __call__(self, graph, examples):
        from torch._dynamo.backends.registry import lookup_backend
        nodes = [str(node.target) for node in graph.graph.nodes if node.op in ("call_function", "call_module", "call_method")
                 and ("lstm" in str(node.target).lower() or
                      node.op == "call_module" and isinstance(graph.get_submodule(node.target), torch.nn.LSTM))]
        if not nodes:
            raise RuntimeError("Full Inductor graph did not capture the LSTM call")
        compiled = lookup_backend("inductor")(graph, examples)
        digest = hashlib.sha256(graph.code.encode()).hexdigest()
        record = None

        def execute(*arguments):
            nonlocal record
            result = compiled(*arguments)
            if record is None:
                # Fullgraph + dynamic=False guards this callable by its static batch size and grad mode.
                # Bind its first successful execution to the real input. Later calls only increment a counter.
                sequence, = [x for x in arguments if isinstance(x, torch.Tensor) and x.ndim == 3
                             and tuple(x.shape[1:]) == (32, 32)]
                batch = sequence.shape[0]
                signature = "train-b64" if torch.is_grad_enabled() and batch == 64 else f"inference-b{batch}"
                record = self.signatures.setdefault(signature, {"graph_sha256": digest, "recurrent_nodes": len(nodes),
                                                               "recurrent_targets": nodes, "executions": 0})
                if record["graph_sha256"] != digest:
                    raise RuntimeError("Unexpected multiple compiled graphs for " + signature)
            record["executions"] += 1
            return result
        return execute

    def evidence(self):
        if set(self.signatures) != {"train-b64", "inference-b1", "inference-b8", "inference-b16"}:
            raise RuntimeError("Missing executed LSTM Inductor signatures")
        return {"backend": "inductor", "dynamic": False, "fullgraph": True, "signatures": self.signatures,
                "dtype": str(torch.get_default_dtype()).removeprefix("torch."),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


class LstmFixture:
    def __init__(self, root, args, device, batches, model):
        self.root = Path(root)
        self.device = device
        self.compilation = ExecutedInductor()
        self.mode = "numeric" if args.lstm_numerics else "timing"
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.receipt = args.lstm_workload_receipt or (args.lstm_numerics / "workload.json" if args.lstm_numerics
                                                     else Path(str(args.output) + ".workload.json"))
        if self.receipt.exists():
            raise FileExistsError(self.receipt)
        provided = dict(epochs=args.epochs, train_batches=args.train_batches, batch_size=args.batch_size,
                        seed=args.seed, inference_batches=batches)
        if (self.manifest["schema"] != "lstm-classification-shared-v1" or self.manifest["model"] != "lstm"
                or provided != PROTOCOL or self.manifest["protocol"] != PROTOCOL or self.manifest["optimizer"] != OPTIMIZER):
            raise ValueError("Mismatched LSTM fixture protocol")
        self.binding = {"fixture_manifest_sha256": file_hash(self.root / "manifest.json"),
                        "initial_state_sha256": file_hash(self.root / "initial.safetensors"),
                        "inputs_sha256": file_hash(self.root / "inputs.safetensors")}
        if any(self.binding[k] != self.manifest[k] for k in ("initial_state_sha256", "inputs_sha256")):
            raise ValueError("Unexpected fixture file digest")
        with TensorFile(self.root / "initial.safetensors") as state:
            model.load_state_dict({k: state.tensor(k) for k in state.names}, strict=True)
        if model.lstm.bias_hh_l0.requires_grad or model.lstm.bias_hh_l0.count_nonzero().item():
            raise ValueError("The second recurrent bias must be fixed at zero")
        self.initial = {k: descriptor(v) for k, v in canonical(model.state_dict()).items()}
        if self.initial != self.manifest["initial"]:
            raise ValueError("Model did not import the fixture weights")
        with TensorFile(self.root / "inputs.safetensors") as stream:
            shapes = input_shapes()
            if stream.names != set(shapes):
                raise ValueError("Unexpected fixture tensors")
            self.values = {}
            for k, shape in shapes.items():
                v = stream.tensor(k)
                if list(v.shape) != shape or not torch.isfinite(v).all():
                    raise ValueError("Invalid " + k)
                if k.endswith("/labels"):
                    if not torch.equal(v.long().float(), v) or (v < 0).any() or (v > 9).any():
                        raise ValueError("Invalid class IDs")
                    v = v.long()
                self.values[k] = v.to(device)
        self.pool = [(self.values[f"train/{i:02d}/input"], self.values[f"train/{i:02d}/labels"]) for i in range(20)]
        self.order = []
        self.inferred = []
        self.observed = self.input_descriptors()
        if self.observed != self.manifest["inputs"]:
            raise ValueError("Observed input descriptors differ")

    def input_descriptors(self):
        return {k: descriptor(v.float()) for k, v in self.values.items()}

    def trained(self, x, y):
        index = len(self.order) % 20
        if self.pool[index][0] is not x or self.pool[index][1] is not y:
            raise ValueError("Training consumed a different batch")
        self.order.append(index)

    def inference(self, batch):
        self.inferred.append(batch)
        return self.values[f"inference/{batch}/input"]

    def complete(self):
        expected = list(range(3)) if self.mode == "numeric" else list(range(20)) * 3
        if self.order != expected or self.inferred != [1, 8, 16] or self.input_descriptors() != self.observed:
            raise ValueError("Incomplete or mutated LSTM workload")
        evidence = dict(schema="lstm-classification-consumption-v1", pid=os.getpid(), model="lstm", framework="pytorch",
                        device=self.device.type, mode=self.mode, initial=self.initial, inputs=self.observed,
                        train_order=self.order, inference_batches=self.inferred, compiler=self.compilation.evidence(),
                        source_sha256=source_hashes(), **self.binding)
        save_json(self.receipt, evidence)

    def numeric(self, original, compiled, output):
        from export_reference import write_safetensors
        output.mkdir(parents=True, exist_ok=True)
        raw = output / "observations.safetensors"
        receipt = output / "numeric-receipt.json"
        if raw.exists() or receipt.exists() or self.receipt.resolve().parent != output.resolve():
            raise ValueError("Use a fresh numeric directory containing its workload receipt")
        params = dict(original.named_parameters())
        optimizer = torch.optim.AdamW(original.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        observed = {}
        compiled.train()
        for i, (x, y) in enumerate(self.pool[:3], 1):
            prefix = f"train/step{i}/"
            optimizer.zero_grad(set_to_none=True)
            logits = compiled(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            observed[prefix + "logits"] = logits.detach().cpu().clone()
            observed[prefix + "loss"] = loss.detach().cpu().clone()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            for k, v in canonical({k: p.grad for k, p in params.items()}).items():
                observed[prefix + "gradient/" + k] = v.detach().cpu().clone()
            optimizer.step()
            self.trained(x, y)
            for k, v in canonical(params).items():
                observed[prefix + "state/" + k] = v.detach().cpu().clone()
        compiled.eval()
        with torch.inference_mode():
            for batch in PROTOCOL["inference_batches"]:
                observed[f"inference/b{batch}/logits"] = compiled(self.inference(batch)).cpu()
        self.complete()
        write_safetensors(raw, observed)
        result = dict(schema="lstm-classification-numerical-v1", pid=os.getpid(), framework="pytorch", device=self.device.type,
                      model="lstm", completed_updates=3, gradient_definition=GRADIENT, source_sha256=source_hashes(),
                      observations={"file": raw.name, "sha256": file_hash(raw)},
                      workload={"file": self.receipt.name, "sha256": file_hash(self.receipt)}, **self.binding)
        save_json(receipt, result)
        return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Export seeded shared inputs and initial weights for the LSTM row")
    cli.add_argument("--output", type=Path, required=True)
    generate(cli.parse_args().output)

