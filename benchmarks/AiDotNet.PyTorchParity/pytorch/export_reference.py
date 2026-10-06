"""Export each PyTorch twin's initial weights plus a step-1 reference for the equivalence check.

For every model this writes two safetensors files into --out:

    <model>.weights.safetensors    the twin's state_dict at initialization (seeded)
    <model>.weights1.safetensors   the state_dict after that one training step
    <model>.reference.safetensors  one fixed batch and what PyTorch computes on it:
        x        the input batch
        y        the integer class labels, as float32
        logits0  forward output at the initial weights
        loss0    cross-entropy of logits0 (a 1-element tensor)
        grad_norm  the global gradient norm before clipping
        logits1  forward output after ONE training step on (x, y): AdamW(lr 1e-3,
                 betas 0.9/0.999, eps 1e-8, weight_decay 0.01), global-norm clip 1.0 --
                 the exact step benchmark.py times

The AiDotNet harness (`--verify-step1`) loads the weights through PyTorchStateDictImporter,
reproduces logits0, takes one Train step and reproduces logits1. Matching logits0 proves the
weights and the forward are the same function; matching logits1 proves the backward and the
optimizer are too. Only then is a timing comparison between the two a comparison of the same work.

Usage:
    python export_reference.py --models mlp,cnn,lstm,transformer --out ../results/reference
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from torch import nn

from benchmark import make_model

REFERENCE_BATCH = 8


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal safetensors writer (float32 only), so the export needs no extra package."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        data = tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)


def export(name: str, out: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model, shape = make_model(name)
    model.train()
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    x = torch.rand((REFERENCE_BATCH, *shape))
    y = torch.randint(0, 10, (REFERENCE_BATCH,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    logits0 = model(x)
    loss0 = criterion(logits0, y)
    optimizer.zero_grad(set_to_none=True)
    loss0.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        logits1 = model(x)

    write_safetensors(out / f"{name}.weights.safetensors", weights)
    write_safetensors(out / f"{name}.weights1.safetensors", {k: v.clone() for k, v in model.state_dict().items()})
    write_safetensors(out / f"{name}.reference.safetensors", {
        "x": x,
        "y": y.to(torch.float32),
        "logits0": logits0,
        "loss0": loss0.reshape(1),
        "grad_norm": grad_norm.reshape(1),
        "logits1": logits1,
    })
    print(f"{name}: {sum(v.numel() for v in weights.values())} weights, loss0={loss0.item():.6f}, "
          f"pre-clip grad norm={grad_norm.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--models", default="mlp,cnn,lstm,transformer")
    parser.add_argument("--out", type=Path, default=Path("../results/reference"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)  # deterministic reduction order for the reference numbers
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        export(name, args.out, args.seed)


if __name__ == "__main__":
    main()
