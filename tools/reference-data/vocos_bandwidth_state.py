"""Extract the complete learned AdaLayerNorm state from the pinned released Vocos checkpoint.

Only the 18 scale/shift tensors are retained. Their full [4,384] shapes and exact float32 values
are preserved; this is component state, not a replacement for the complete pretrained vocoder.
Usage: python vocos_bandwidth_state.py --checkpoint path/to/pytorch_model.bin [--write]
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

REVISION = "4e61d082c08045a4c11e5b148ad93b1d0c591a14"
CHECKPOINT_SHA256 = "7e95bb260b74a1bfc43c52d355831c951acb81c8960e9c62b79bd2b3ab1e3a90"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("Released checkpoint SHA-256 mismatch")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    names = [f"{prefix}.{part}.weight"
             for prefix in ["backbone.norm"] + [f"backbone.convnext.{i}.norm" for i in range(8)]
             for part in ["scale", "shift"]]
    tensors = []
    for name in names:
        value = state[name]
        if list(value.shape) != [4, 384] or value.dtype != torch.float32:
            raise RuntimeError(f"Unexpected released tensor {name}: {value.shape}/{value.dtype}")
        tensors.append(dict(name=name, shape=list(value.shape), values=value.flatten().tolist()))
    result = dict(model="charactr/vocos-encodec-24khz", revision=REVISION,
                  checkpoint_sha256=CHECKPOINT_SHA256, tensors=tensors)
    expected = (json.dumps(result, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    path = Path(__file__).resolve().parents[2] / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/vocos_released_bandwidth_state.json"
    if args.write:
        path.write_bytes(expected)
    elif path.read_bytes() != expected:
        raise RuntimeError("Extracted released state differs from the committed artifact")
    print(f"18 complete released tensors; artifact SHA-256 {hashlib.sha256(expected).hexdigest()}")


if __name__ == "__main__":
    main()
