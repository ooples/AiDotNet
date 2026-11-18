"""Export full HiFi-GAN V1 correctness evidence, separate from timing measurements.

Reuses benchmark.py's actual V1 networks, mel analysis, losses and AdamW train_step. The
reference includes identical initial learned weights and spectral state, three fixed 32-frame
crops, isolated D/G gradients, G loss components and all common state after three real updates.
No reduced-width model or synthetic substitute for the shipped training step is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

from benchmark import HifiGanV1, HifiMel, HIFI_HOP, HIFI_MELS, HIFI_RATE, HIFI_SEGMENT
from export_reference import write_safetensors


def ordered_layers(model: HifiGanV1):
    """The shipped C# builder's order: pre, each upsample and its MRFs, post, periods, scales."""
    generator = model.generator
    yield "generator/000", "generator.conv_pre", generator.conv_pre
    index = 1
    for stage, upsample in enumerate(generator.ups):
        yield f"generator/{index:03}", f"generator.ups.{stage}", upsample
        index += 1
        for branch in range(3):
            block_index = stage * 3 + branch
            block = generator.resblocks[block_index]
            for bank in ("convs1", "convs2"):
                for dilation, conv in enumerate(getattr(block, bank)):
                    yield f"generator/{index:03}", f"generator.resblocks.{block_index}.{bank}.{dilation}", conv
                    index += 1
    yield f"generator/{index:03}", "generator.conv_post", generator.conv_post
    index = 0
    for family in ("periods", "scales"):
        for critic_index, critic in enumerate(getattr(model.discriminators, family)):
            for conv_index, conv in enumerate(critic.convs):
                yield f"discriminator/{index:03}", f"discriminators.{family}.{critic_index}.convs.{conv_index}", conv
                index += 1
            yield f"discriminator/{index:03}", f"discriminators.{family}.{critic_index}.conv_post", critic.conv_post
            index += 1


def fields(conv):
    parametrization = conv.parametrizations.weight
    if hasattr(parametrization, "original1"):
        yield "direction", parametrization.original1
        yield "gain", parametrization.original0
    else:
        yield "direction", parametrization.original
        yield "u", parametrization[0]._u
        yield "v", parametrization[0]._v
    if conv.bias is not None:
        yield "bias", conv.bias


def logical(tensor: torch.Tensor, role: str) -> torch.Tensor:
    if role == "direction":
        # Period Conv2d has [out,in,k,1]; C# stores its identical flattened weights as [out,in,1,k].
        return tensor.reshape(tensor.shape[0], tensor.shape[1], 1, -1)
    return tensor.reshape(-1)


def state(model: HifiGanV1) -> dict[str, torch.Tensor]:
    return {f"{prefix}/{role}": logical(value.detach(), role).clone()
            for prefix, _, conv in ordered_layers(model) for role, value in fields(conv)}


def gradients(model: HifiGanV1, family: str) -> dict[str, torch.Tensor]:
    result = {}
    for prefix, _, conv in ordered_layers(model):
        if not prefix.startswith(family + "/"):
            continue
        for role, value in fields(conv):
            if role in ("u", "v"):
                continue
            if value.grad is None:
                raise RuntimeError(f"Missing gradient: {prefix}/{role}")
            result[f"{prefix}/{role}"] = logical(value.grad.detach(), role).clone()
    return result


def generator_terms(model: HifiGanV1, mel: torch.Tensor, real: torch.Tensor):
    generated = model.generator(mel)
    fake_out = model.discriminators(generated)
    with torch.no_grad():
        real_out = model.discriminators(real)
        real_mel = model.loss_mel(real.squeeze(1))
    adversarial = sum(torch.mean((1 - score) ** 2) for score, _ in fake_out)
    matching = sum(torch.mean(torch.abs(r - g)) for (_, rf), (_, gf) in zip(real_out, fake_out)
                   for r, g in zip(rf, gf))
    mel_loss = F.l1_loss(model.loss_mel(generated.squeeze(1)), real_mel)
    return generated, adversarial, matching, mel_loss, adversarial + 2 * matching + 45 * mel_loss


def batch(step: int, device: torch.device, frames: int = 32):
    time = torch.arange(frames * HIFI_HOP, dtype=torch.float64) / HIFI_RATE
    audio = (.5 * torch.sin(2 * torch.pi * (220 + 17 * step) * time)
             + .2 * torch.sin(2 * torch.pi * (437 + 13 * step) * time + .3)
             + .03 * torch.cos(time * 11213)).to(torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mel = HifiMel(8000.0).to(device)(audio)
    if tuple(mel.shape) != (1, HIFI_MELS, frames):
        raise RuntimeError(f"The fixed crop has unexpected geometry: {tuple(mel.shape)}")
    return mel, audio


def unsupported_determinism_warnings(warnings_seen):
    return [row for row in warnings_seen if "deterministic" in row["message"].lower()
            and any(text in row["message"].lower() for text in
                    ("does not have", "nondeterministic", "non-deterministic"))]


def write_warning_receipt(out: Path, device: torch.device, warnings_seen, completed: bool):
    receipt = {"schema": 1, "device": str(device), "reference_completed": completed,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
               "bitwise_determinism_claim": False,
               "unsupported_determinism_warnings": unsupported_determinism_warnings(warnings_seen),
               "warnings": warnings_seen,
               "exporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = out / "determinism-warnings.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(out: Path, device: torch.device, seed: int, compile_model: bool, workload_root: Path | None = None):
    warnings_seen = []
    completed = False
    with warnings.catch_warnings():
        previous_showwarning = warnings.showwarning
        def observe(message, category, filename, lineno, file=None, line=None):
            warnings_seen.append({"message": str(message), "category": category.__name__,
                                  "filename": filename, "lineno": lineno})
            previous_showwarning(message, category, filename, lineno, file, line)
        warnings.showwarning = observe
        try:
            _export(out, device, seed, compile_model, warnings_seen, workload_root)
            completed = True
        finally:
            write_warning_receipt(out, device, warnings_seen, completed)


def _export(out: Path, device: torch.device, seed: int, compile_model: bool, warnings_seen, workload_root: Path | None = None):
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    # CUDA reflection-pad backward has no deterministic implementation. Keep deterministic
    # choices where available and report unsupported kernels rather than changing the math.
    torch.use_deterministic_algorithms(True, warn_only=device.type == "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = HifiGanV1().to(device)
    workload = None
    imported404 = None
    if workload_root is not None:
        from hifigan_fixture import HifiBenchmarkFixture
        workload = HifiBenchmarkFixture(str(workload_root), epochs=3, batches=3, batch_size=1, seed=seed,
                                        inference_batches=[1, 8, 16], receipt=str(out / "unused-benchmark-consumption.json"), device=device)
        imported404 = workload.import_state(model)

    if sum(parameter.numel() for parameter in model.parameters()) != 84_660_721:
        raise RuntimeError("The reference must preserve all84,660,721 full-V1 learned parameters.")
    if compile_model:
        model.compile_parts()
    layers = list(ordered_layers(model))
    if sum(p.startswith("generator/") for p, _, _ in layers) != 78 or len(layers) != 132:
        raise RuntimeError("The reference is not the full V1 layer geometry.")
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    write_safetensors(out / "initial.safetensors", state(model))
    reference: dict[str, torch.Tensor] = {}
    if workload is None:
        batches = [batch(step, device) for step in range(3)]
    else:
        batches = []
        for record in workload.steps[:3]:
            source_mel, source_audio = workload.pool[record["pool_index"]]
            start = record["crop_start"]
            batches.append((source_mel[:, :, start:start + 32].contiguous(),
                            source_audio[:, start * HIFI_HOP:(start + 32) * HIFI_HOP].contiguous()))
    for step, (mel, audio) in enumerate(batches):
        reference[f"mel/{step}"] = mel
        reference[f"audio/{step}"] = audio.reshape(-1)

    mel, audio = batches[0]
    model.eval()
    with torch.no_grad():
        reference["generator/output0"] = model.generator(mel).clone()
        inference_mel = batch(19, device, frames=256)[0] if workload is None else None
        for count in (1, 8, 16):
            if workload is None:
                inputs = inference_mel.expand(count, -1, -1).contiguous()
                inputs = inputs + torch.arange(count, device=device).reshape(-1, 1, 1) * .001
            else:
                inputs = workload.input_values[f"inference/{count}/mel"]
            reference[f"inference/mel/{count}"] = inputs
            reference[f"inference/output/{count}"] = model.generator(inputs).clone()
    model.train()
    with torch.no_grad():
        fake = model.generator(mel)
    discriminator_loss = sum(torch.mean((1 - r) ** 2) + torch.mean(g ** 2)
                             for (r, _), (g, _) in zip(model.discriminators(audio.unsqueeze(1)),
                                                      model.discriminators(fake)))
    model.zero_grad(set_to_none=True)
    discriminator_loss.backward()
    isolated = gradients(model, "discriminator")
    reference["isolated/discriminator_loss"] = discriminator_loss.detach().reshape(1)

    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    model.discriminators.requires_grad_(False)
    generated, adversarial, matching, mel_loss, generator_loss = generator_terms(model, mel, audio.unsqueeze(1))
    generator_loss.backward()
    isolated.update(gradients(model, "generator"))
    model.discriminators.requires_grad_(True)
    for key, value in (("adversarial", adversarial), ("feature_matching", matching),
                       ("mel_l1", mel_loss), ("generator_loss", generator_loss)):
        reference["isolated/" + key] = value.detach().reshape(1)
    write_safetensors(out / "isolated-gradients.safetensors", isolated)
    del isolated, generated, discriminator_loss, adversarial, matching, mel_loss, generator_loss

    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    for step, (mel, audio) in enumerate(batches):
        loss = model.train_step(mel, audio)
        reference[f"train/generator_loss/{step}"] = loss.reshape(1)
        write_safetensors(out / f"step{step + 1}.safetensors", state(model))
    write_safetensors(out / "reference.safetensors", reference)
    names = ("initial", "isolated-gradients", "step1", "step2", "step3", "reference")
    files = {name + ".safetensors": hashlib.sha256((out / (name + ".safetensors")).read_bytes()).hexdigest() for name in names}
    warning_receipt_hash = write_warning_receipt(out, device, warnings_seen, completed=True)
    manifest = {
        "schema": "hifigan-v1-equivalence-v1", "model": "hifigan-v1", "device": str(device),
        "torch_version": torch.__version__, "compiled": compile_model, "seed": seed, "steps": 3, "crop_start_frame": 0,
        "geometry": {"mel_channels": 80, "frames": 32, "samples": 8192,
                     "inference_frames": 256, "inference_batches": [1, 8, 16],
                     "generator_layers": 78, "discriminator_layers": 54},
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": {"type": "AdamW", "lr": 2e-4, "betas": [.8, .99], "eps": 1e-8, "weight_decay": .01},
        "tf32": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "bitwise_determinism_claim": False,
        "unsupported_determinism_warnings": unsupported_determinism_warnings(warnings_seen),
        "determinism_warning_receipt": {"file": "determinism-warnings.json", "sha256": warning_receipt_hash},
        "exporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files_sha256": files,
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).with_name("benchmark.py").read_bytes()).hexdigest(),
        "layers": [{"key": prefix, "torch_name": name,
                    "normalization": "spectral" if not hasattr(conv.parametrizations.weight, "original1") else "weight",
                    "native_weight_shape": list(conv.parametrizations.weight.original.shape)
                    if not hasattr(conv.parametrizations.weight, "original1")
                    else list(conv.parametrizations.weight.original1.shape)} for prefix, name, conv in layers],
        "scopes": ["generator_eval_output_b1_b8_b16_at256_frames", "isolated_D_loss_and_all_gradients",
                   "isolated_G_loss_components_and_all_gradients", "three_actual_train_updates_and_spectral_state"],
    }

    manifest["recipe_artifacts_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("benchmark.py", "benchmark_telemetry.py", "export_reference.py", "export_hifigan_reference.py", "hifigan_fixture.py", "export_hifigan_workload.py")}
    if workload is not None:
        manifest["workload_manifest_sha256"] = workload.manifest_sha256
        manifest["benchmark_crops_before_fixed32_reference_input"] = workload.actual_steps[:3]
        manifest["workload_initial_state_sha256"] = workload.manifest["state_sha256"]
        (out / "reference-workload-inputs.json").write_text(json.dumps({
            "schema": "hifigan-matched-reference-inputs-v1", "fixture_manifest_sha256": workload.manifest_sha256,
            "actual_imported404": imported404, "first_three_training_steps": workload.actual_steps[:3],
            "actual_inputs": workload.actual_inputs, "benchmark_consumption_claim": False,
        }, indent=2) + "\n", encoding="utf-8")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported": str(out), "device": str(device), "parameters": manifest["trainable_parameters"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eager", action="store_true", help="Diagnostic reference only; the completion verifier requires torch.compile.")
    parser.add_argument("--workload-fixture", type=Path)

    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA correctness reference requested but CUDA is unavailable.")
    args.out.mkdir(parents=True, exist_ok=True)
    export(args.out, torch.device(args.device), args.seed, compile_model=not args.eager,
           workload_root=args.workload_fixture)


if __name__ == "__main__":
    main()
