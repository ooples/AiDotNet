"""Prepare immutable shared404-state/full-utterance data for BOTH real benchmark paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch

from benchmark import HifiGanV1
from export_reference import write_safetensors
from hifigan_fixture import TensorFile, descriptor, file_hash, tensor_bytes


def export(reference: Path, output: Path, seed: int):
    if output.exists():
        raise FileExistsError("Use a fresh immutable workload directory.")
    reference_manifest = reference / "manifest.json"
    authority = json.loads(reference_manifest.read_text(encoding="utf-8"))
    initial = reference / "initial.safetensors"
    if file_hash(initial) != authority["files_sha256"]["initial.safetensors"]:
        raise ValueError("Canonical reference state hash differs.")
    if authority["trainable_parameters"] != 84_660_721:
        raise ValueError("Full V1 reference state required.")
    if authority["seed"] != seed:
        raise ValueError("The canonical initial-state seed differs from the workload seed.")
    output.mkdir(parents=True)
    shutil.copyfile(initial, output / "initial.safetensors")
    state, digest = {}, hashlib.sha256()
    with TensorFile(output / "initial.safetensors") as source:
        if len(source.names) != 404:
            raise ValueError("Exactly388 learned fields plus16 spectral buffers required.")
        for name in sorted(source.names):
            value = source.tensor(name)
            entry = descriptor(value)
            state[name] = entry
            digest.update((name + "|" + ",".join(map(str, entry["shape"])) + "|").encode())
            digest.update(tensor_bytes(value))
    learned = [name for name in state if name.rsplit("/", 1)[1] in ("direction", "gain", "bias")]
    spectral = [name for name in state if name.rsplit("/", 1)[1] in ("u", "v")]
    elements = 0
    for name in learned:
        count = 1
        for dim in state[name]["shape"]:
            count *= dim
        elements += count
    if len(learned) != 388 or len(spectral) != 16 or elements != 84_660_721:
        raise ValueError("Canonical learned/spectral inventory differs.")
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    inputs = {}
    for index in range(3):
        mel, audio = HifiGanV1.make_batch(1, torch.device("cpu"))
        inputs[f"train/{index}/mel"] = mel
        inputs[f"train/{index}/audio"] = audio
    for batch in (1, 8, 16):
        mel, _ = HifiGanV1.make_batch(batch, torch.device("cpu"))
        inputs[f"inference/{batch}/mel"] = mel
    input_descriptors = {name: descriptor(value) for name, value in sorted(inputs.items())}
    write_safetensors(output / "inputs.safetensors", inputs)
    crop_random = random.Random(seed)
    steps = []
    for index in range(9):
        pool = index % 3
        start = crop_random.randint(0, 224)
        mel, audio = inputs[f"train/{pool}/mel"], inputs[f"train/{pool}/audio"]
        steps.append({"index": index, "epoch": index // 3, "pool_index": pool, "crop_start": start,
                      "mel_key": f"train/{pool}/mel", "audio_key": f"train/{pool}/audio",
                      "mel": input_descriptors[f"train/{pool}/mel"],
                      "audio": input_descriptors[f"train/{pool}/audio"],
                      "cropped_mel": descriptor(mel[:, :, start:start + 32]),
                      "cropped_audio": descriptor(audio[:, start * 256:(start + 32) * 256])})
    manifest = {
        "schema": "hifigan-shared-workload-v1",
        "protocol": {"epochs": 3, "train_batches": 3, "batch_size": 1, "seed": seed,
                     "inference_batches": [1, 8, 16], "utterance_frames": 256,
                     "segment_frames": 32, "hop": 256, "mel_channels": 80},
        "learned_parameters": 84_660_721, "state_tensor_count": 404,
        "layers": {"generator": 78, "discriminator": 54},
        "state_sha256": digest.hexdigest(), "state": state, "inputs": input_descriptors,
        "training_steps": steps,
        "construction_warmup": {"input_key": "inference/1/mel", "mode": "eval_no_grad",
                                "state_after_warmup": "explicit_restore_then_actual404_hash_before_training"},
        "files_sha256": {name: file_hash(output / name) for name in ("initial.safetensors", "inputs.safetensors")},
        "reference_manifest_sha256": file_hash(reference_manifest),
        "benchmark_source_sha256": file_hash(Path(__file__).with_name("benchmark.py")),
        "exporter_sha256": file_hash(Path(__file__)),
        "fixture_consumer_sha256": file_hash(Path(__file__).with_name("hifigan_fixture.py")),
        "identity_contract": "F32 little-endian payload SHA256; aggregate sorted ASCII name|comma-separated shape| then raw bytes. Three full utterance batches replayed for nine steps; fixed crop providers validate every actual bound/index. No RNG-generated measured inputs.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    result = export(args.reference.resolve(), args.output.resolve(), args.seed)
    print(json.dumps({"schema": result["schema"], "state_fields": result["state_tensor_count"],
                      "training_steps": len(result["training_steps"]), "state_sha256": result["state_sha256"],
                      "manifest_sha256": file_hash(args.output / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
