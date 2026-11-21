"""Hash-bound shared inputs/state for the real HiFi benchmark; no model math lives here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


READBACK_ELEMENTS = 262144  # At most1MiB per GPU-to-host observation chunk.


def tensor_bytes(value) -> memoryview:
    """CPU-contiguous read-only bytes without a field-sized identity-observer copy."""
    value = value.detach()
    if value.device.type != "cpu" or not value.is_contiguous():
        raise ValueError("A CPU contiguous view is required; use the bounded chunk observer otherwise.")
    return memoryview(value.numpy()).cast("B").toreadonly()


class TensorHashReader:
    def __init__(self):
        self.host = None

    def chunks(self, value):
        import torch
        value = value.detach()
        if not value.is_contiguous():
            if value.numel() > READBACK_ELEMENTS:
                raise ValueError("An unbounded noncontiguous identity-observer copy is forbidden.")
            value = value.contiguous()  # Only small cropped-input scratch; never a whole weight field.
        flat = value.reshape(-1)
        if flat.device.type == "cpu":
            view = tensor_bytes(flat)
            for offset in range(0, len(view), READBACK_ELEMENTS * 4):
                yield view[offset:offset + READBACK_ELEMENTS * 4]
        else:
            if self.host is None:
                self.host = torch.empty(READBACK_ELEMENTS, dtype=torch.float32, device="cpu")
            for offset in range(0, flat.numel(), READBACK_ELEMENTS):
                count = min(READBACK_ELEMENTS, flat.numel() - offset)
                host = self.host[:count]
                host.copy_(flat[offset:offset + count])
                yield tensor_bytes(host)


def descriptor(value, reader=None) -> dict:
    import torch
    if value.dtype != torch.float32:
        raise ValueError("The shared workload permits only explicit FP32 tensors.")
    digest = hashlib.sha256()
    for chunk in (reader or TensorHashReader()).chunks(value):
        digest.update(chunk)
    return {"dtype": "F32", "shape": list(value.shape), "sha256": digest.hexdigest()}


def aggregate(descriptors: dict[str, dict], payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(descriptors):
        entry = descriptors[name]
        digest.update((name + "|" + ",".join(str(dim) for dim in entry["shape"]) + "|").encode("utf-8"))
        digest.update(payloads[name])
    return digest.hexdigest()


class TensorFile:
    """One tensor-sized buffer at a time; no full checkpoint mapping survives state import."""
    def __init__(self, path: Path):
        if sys.byteorder != "little":
            raise RuntimeError("The fixture's FP32 byte contract is little endian.")
        stream = path.open("rb")
        try:
            self._read_header(stream, path)
        except BaseException:
            # __exit__ never runs for a constructor that raised; do not leave the handle to the traceback.
            stream.close()
            raise
        self.stream = stream

    def _read_header(self, stream, path: Path):
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors length.")
        length, = struct.unpack("<Q", prefix)
        if length > 16 * 1024 * 1024:
            raise ValueError("Unbounded safetensors metadata.")
        self.entries = json.loads(stream.read(length))
        self.origin = 8 + length
        size = path.stat().st_size - self.origin
        for name, entry in self.entries.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            elements = 1
            for dim in entry["shape"]:
                if not isinstance(dim, int) or dim < 0:
                    raise ValueError("Invalid fixture shape.")
                elements *= dim
            if entry["dtype"] != "F32" or begin < 0 or end < begin or end > size or end - begin != elements * 4:
                raise ValueError("Invalid FP32 fixture payload: " + name)

    @property
    def names(self):
        return set(self.entries) - {"__metadata__"}

    def tensor(self, name: str):
        import torch
        entry = self.entries[name]
        begin, end = entry["data_offsets"]
        payload = bytearray(end - begin)
        self.stream.seek(self.origin + begin)
        if self.stream.readinto(payload) != len(payload):
            raise ValueError("Truncated tensor: " + name)
        return torch.frombuffer(payload, dtype=torch.float32).reshape(entry["shape"])

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HifiBenchmarkFixture:
    def __init__(self, root: str, *, epochs: int, batches: int, batch_size: int, seed: int,
                 inference_batches: list[int], receipt: str, device):
        self.root = Path(root).resolve()
        self.path = self.root / "manifest.json"
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.manifest_sha256 = file_hash(self.path)
        self.receipt = Path(receipt).resolve()
        if self.receipt.exists():
            raise FileExistsError("Preserve prior workload identity evidence: " + str(self.receipt))
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        expected = {"epochs": epochs, "train_batches": batches, "batch_size": batch_size,
                    "seed": seed, "inference_batches": inference_batches, "utterance_frames": 256,
                    "segment_frames": 32, "hop": 256, "mel_channels": 80}
        if self.manifest.get("schema") != "hifigan-shared-workload-v1" or self.manifest.get("protocol") != expected:
            raise ValueError("Fixture protocol differs from the actual benchmark invocation.")
        if self.manifest.get("benchmark_source_sha256") != file_hash(Path(__file__).with_name("benchmark.py")) or self.manifest.get("fixture_consumer_sha256") != file_hash(Path(__file__)):
            raise ValueError("The fixture is bound to different actual benchmark/consumer source.")
        if (epochs, batches, batch_size, inference_batches) != (3, 3, 1, [1, 8, 16]):
            raise ValueError("This initial shared workload requires the explicit3x3/b1,b8,b16 protocol.")
        if self.manifest.get("state_tensor_count") != 404 or self.manifest.get("learned_parameters") != 84_660_721:
            raise ValueError("Full shared404-field/84,660,721-parameter state required.")
        if set(self.manifest["files_sha256"]) != {"initial.safetensors", "inputs.safetensors"}:
            raise ValueError("Every fixture data file must be hash-bound exactly once.")
        for name, expected_hash in self.manifest["files_sha256"].items():
            if file_hash(self.root / name) != expected_hash:
                raise ValueError("Fixture file changed: " + name)
        self.steps = self.manifest["training_steps"]
        if len(self.steps) != 9 or [step["index"] for step in self.steps] != list(range(9)):
            raise ValueError("Exactly nine ordered training records required.")
        if [step["pool_index"] for step in self.steps] != [0, 1, 2] * 3:
            raise ValueError("The actual three-batch pool must repeat in the same epoch order.")
        names = {f"train/{index}/{role}" for index in range(3) for role in ("mel", "audio")}
        names.update(f"inference/{batch}/mel" for batch in (1, 8, 16))
        if set(self.manifest["inputs"]) != names:
            raise ValueError("Exactly the actual three training pairs and three inference inputs are required.")
        self.input_values = {}
        self.actual_inputs = {}
        reader = TensorHashReader()
        with TensorFile(self.root / "inputs.safetensors") as source:
            if source.names != set(self.manifest["inputs"]):
                raise ValueError("Missing or extra shared input tensors.")
            for name in sorted(source.names):
                value = source.tensor(name).to(device=device)
                actual = descriptor(value, reader)
                if actual != self.manifest["inputs"][name]:
                    raise ValueError("Actual loaded input differs: " + name)
                self.input_values[name] = value
                self.actual_inputs[name] = actual
        self.pool = [(self.input_values[f"train/{index}/mel"], self.input_values[f"train/{index}/audio"])
                     for index in range(3)]
        for index, (mel, audio) in enumerate(self.pool):
            if tuple(mel.shape) != (1, 80, 256) or tuple(audio.shape) != (1, 65536):
                raise ValueError("Full training utterance geometry required.")
        for batch in [1, 8, 16]:
            if tuple(self.input_values[f"inference/{batch}/mel"].shape) != (batch, 80, 256):
                raise ValueError("Full inference geometry required.")
        self.actual_steps = []
        for step in self.steps:
            start = step["crop_start"]
            if not isinstance(start, int) or not 0 <= start <= 224:
                raise ValueError("Invalid canonical crop.")
            mel, audio = self.pool[step["pool_index"]]
            actual_mel = descriptor(mel[:, :, start:start + 32], reader)
            actual_audio = descriptor(audio[:, start * 256:(start + 32) * 256], reader)
            if actual_mel != step["cropped_mel"] or actual_audio != step["cropped_audio"] or step["mel"] != self.actual_inputs[step["mel_key"]] or step["audio"] != self.actual_inputs[step["audio_key"]]:
                raise ValueError("Actual loaded cropped training input differs.")
            self.actual_steps.append({**step, "cropped_mel": actual_mel, "cropped_audio": actual_audio})
        self.consumed_count = 0
        self.inference_consumed = []
        self.preflight = None

    def import_state(self, model):
        import torch
        from export_hifigan_reference import ordered_layers, fields, logical
        actual = {}
        learned = 0
        targets = {prefix + "/" + role: (target, role)
                   for prefix, _, conv in ordered_layers(model) for role, target in fields(conv)}
        digest = hashlib.sha256()
        reader = TensorHashReader()
        with TensorFile(self.root / "initial.safetensors") as source, torch.no_grad():
            if set(targets) != source.names or source.names != set(self.manifest["state"]) or len(source.names) != 404:
                raise ValueError("Missing/extra active shared state.")
            for name in sorted(targets):
                target, role = targets[name]
                value = source.tensor(name)
                expected = self.manifest["state"][name]
                if descriptor(value) != expected or value.numel() != target.numel():
                    raise ValueError("Canonical state/shape differs: " + name)
                target.copy_(value.reshape(target.shape))
                version = target._version
                stored = logical(target.detach(), role)
                digest.update((name + "|" + ",".join(map(str, stored.shape)) + "|").encode())
                field_digest = hashlib.sha256()
                for chunk in reader.chunks(stored):
                    field_digest.update(chunk)
                    digest.update(chunk)
                actual[name] = {"dtype": "F32", "shape": list(stored.shape),
                                "sha256": field_digest.hexdigest()}
                if actual[name] != expected:
                    raise ValueError("Actual imported404 field differs: " + name)
                if target._version != version:
                    raise ValueError("The state identity observer mutated its source.")
                if role in ("direction", "gain", "bias"):
                    learned += target.numel()
            if learned != 84_660_721:
                raise ValueError("Full learned parameter inventory differs.")
        if digest.hexdigest() != self.manifest["state_sha256"]:
            raise ValueError("Actual shared404 aggregate differs.")
        return actual

    def next_crop(self, low: int, high: int, mel, audio) -> int:
        index = self.consumed_count
        if low != 0 or high != 224 or index >= 9:
            raise ValueError("Unexpected actual crop bound or extra training step.")
        step = self.steps[index]
        expected_mel, expected_audio = self.pool[step["pool_index"]]
        if mel is not expected_mel or audio is not expected_audio:
            raise ValueError("The actual training step received a different prevalidated batch.")
        self.consumed_count += 1
        return step["crop_start"]

    def inference(self, batch: int):
        index = len(self.inference_consumed)
        if index >= 3 or [1, 8, 16][index] != batch:
            raise ValueError("Unexpected actual inference batch order.")
        self.inference_consumed.append(batch)
        return self.input_values[f"inference/{batch}/mel"]

    def emit_preflight(self, model):
        # Called after the same explicit warmup on both frameworks. Deliberately restore canonical state again.
        actual = self.import_state(model)
        self.preflight = {"schema": "hifigan-shared-workload-consumption-v1", "framework": "pytorch",
                          "pid": os.getpid(), "state_tensor_count": len(actual),
                          "pass": False, "preflight_pass": True, "fixture_manifest_sha256": self.manifest_sha256,
                          "state_sha256": self.manifest["state_sha256"], "state": actual,
                          "inputs": self.actual_inputs, "training_steps": self.actual_steps,
                          "consumed_steps": [], "inference_batches_consumed": [],
                          "state_checkpoint_retained": False,
                          "hash_observer": {"cpu_payload_clone": False, "gpu_chunk_bytes_max": READBACK_ELEMENTS * 4,
                                            "gpu_scratch_reused_within_observation": True,
                                            "gpu_scratch_retained_during_measurement": False,
                                            "source_version_unchanged": True}}
        self.receipt.write_text(json.dumps(self.preflight, indent=2) + "\n", encoding="utf-8")

    def complete(self):
        if self.preflight is None or self.consumed_count != 9 or self.inference_consumed != [1, 8, 16]:
            raise ValueError("The actual benchmark did not consume the complete canonical workload.")
        reader = TensorHashReader()
        for name, value in self.input_values.items():
            if descriptor(value, reader) != self.manifest["inputs"][name]:
                raise ValueError("The actual training/inference mutated shared data: " + name)
        self.preflight["pass"] = True
        self.preflight["consumed_steps"] = [{key: step[key] for key in ("index", "pool_index", "crop_start")}
                                            for step in self.steps]
        self.preflight["inference_batches_consumed"] = self.inference_consumed
        self.receipt.write_text(json.dumps(self.preflight, indent=2) + "\n", encoding="utf-8")
