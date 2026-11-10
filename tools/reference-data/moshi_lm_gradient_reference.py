"""Full Moshi LM derivatives from the pinned, unmodified author's LMModel.

Uses the existing actual author state and Eq. 7 inputs. The same explicit
float64 RMSNorm/RoPE precision alignment as moshi_lm_reference.py is retained.
No native implementation participates in reference generation.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

PIN = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_lm_reference.json"
OUTPUT = BASE.with_name("moshi_lm_gradient_reference.json")


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def eq7(torch, out, codes, streams):
    text = codes[0, 0]
    weights = out.text_mask[0, 0].double() * torch.where((text == 3) | (text == 0), 0.5, 1.0)
    ce = torch.nn.functional.cross_entropy(torch.nan_to_num(out.text_logits[0, 0]), text.clamp(min=0), reduction="none")
    text_loss = (ce * weights).sum() / weights.sum()
    alpha = torch.tensor([100.0 if s % 3 == 0 else 1.0 for s in range(streams)]).view(streams, 1)
    weights = out.mask[0].double() * alpha
    logits = torch.nan_to_num(out.logits[0])
    targets = codes[0, 1:1 + streams].clamp(min=0)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), targets.reshape(-1), reduction="none").view(streams, -1)
    return text_loss + (ce * weights).sum() / weights.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    if git(source, "rev-parse", "HEAD") != PIN:
        raise RuntimeError(f"Moshi source must be pinned to {PIN}")
    sys.path.insert(0, str(source / "moshi"))
    import torch
    import safetensors
    from safetensors.torch import save
    import moshi_lm_reference as reference

    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    if not (2, 2) <= version < (2, 10):
        raise RuntimeError("The pinned author package requires torch >=2.2,<2.10")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    reference.patch_float64()
    original = json.loads(BASE.read_text(encoding="utf-8"))
    codes = torch.tensor([original["codes"]])
    cases = {}
    for name, streams, seed in (("paper", 6, 2294), ("release", 3, 2295)):
        model = reference.build(streams, seed, original, f"{name}_state")
        out = model(codes)
        loss = eq7(torch, out, codes, streams)
        if name == "paper" and abs(loss.item() - original["paper_loss"]) > 1e-10:
            raise RuntimeError("Supported author runtime no longer reproduces the existing Eq. 7 loss")
        loss.backward()
        parameters = dict(model.named_parameters())
        unused = [name for name, value in parameters.items() if value.grad is None]
        gradients = {name: (torch.zeros_like(value) if value.grad is None else value.grad).contiguous()
                     for name, value in parameters.items()}
        if any(not torch.isfinite(value).all() for value in gradients.values()):
            raise RuntimeError("Nonfinite author derivative")
        cases[name] = {
            "loss": loss.item(), "parameter_tensors": len(parameters),
            "parameter_scalars": sum(value.numel() for value in parameters.values()),
            "parameter_shapes": {name: list(value.shape) for name, value in parameters.items()},
            "unused_parameters": unused,
            "nonzero_gradient_tensors": sum(bool(torch.count_nonzero(value)) for value in gradients.values()),
            "gradient_norm": sum(value.square().sum().item() for value in gradients.values()) ** 0.5,
            "gradient_state": base64.b64encode(save(gradients)).decode("ascii"),
        }

    # Check every imported author Python file against the pinned Git tree, not
    # just a package name/version or a locally edited checkout's HEAD.
    files = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(source):
            continue
        relative = path.relative_to(source).as_posix()
        content = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        expected = git(source, "rev-parse", f"{PIN}:{relative}")
        if actual != expected:
            raise RuntimeError(f"Author source was modified: {relative}")
        files[relative] = actual
    result = {
        "generator": "tools/reference-data/moshi_lm_gradient_reference.py",
        "author_commit": PIN, "author_files": dict(sorted(files.items())),
        "python": platform.python_version(), "torch": torch.__version__, "safetensors": safetensors.__version__,
        "precision_alignment": "Explicit float64 RMSNorm and RoPE from moshi_lm_reference.py; all other author definitions unchanged",
        "base_fixture_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(), "cases": cases,
    }
    if args.write:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        stored = json.loads(args.output.read_text(encoding="utf-8"))
        if stored != result:
            raise RuntimeError("Moshi derivative reference differs; investigate before regenerating")
    print({name: {key: case[key] for key in ("parameter_tensors", "parameter_scalars", "unused_parameters", "gradient_norm")}
           for name, case in cases.items()})


if __name__ == "__main__":
    main()
