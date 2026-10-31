"""Reference fixture for GPT-SoVITS's flow-matching synthesizer (module/models.py SynthesizerTrnV3, CFM, models_v5.CFMV5)
for v3, v4 and v5dev.

Runs the release's own modules (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image in float64 with small sizes:
the DiT is built through the release's constructor call with dim 32, depth 2, 2 heads of 16; the widths the release fixes
(SSL 768, MRTE 512, the 512-channel bridge and wns1) are replaced after construction by modules of the same kind with
smaller widths. Every torch.randn the CFM draws is replaced by a closed-form pattern (noise below) so the fixture need
not store it. Records per version: the state dict; the style vector; decode_encp at speed 1 and 1.25 (with the recorded
style vector); one cfm.inference; the release's chunked synthesis (inference_webui's v3/v4 loop, synthesize_v5_mel) over
a target long enough for two chunks, subsampled; the training loss with its draws fixed (with and without the
shortcut bootstrap for v3/v4); and for v3, LoRA adapters (peft, as s2_train_v3_lora.py attaches them) before and after
merging.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_cfm_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_cfm_reference.json.
"""
import json
import math
import os
import random
import sys

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ.setdefault("version", "v2")

from module import models  # noqa: E402
from module import models_v5  # noqa: E402
from module.models_v5 import synthesize_v5_mel  # noqa: E402
from module.modules import MelStyleEncoder  # noqa: E402
from module.mrte_model import MRTE  # noqa: E402
from module.quantize import ResidualVectorQuantizer  # noqa: E402

SSL = 24
HIDDEN = 16
GIN = 16
BRIDGE = 32
STRIDE = 7
real_dit = models.DiT
models.DiT = lambda **kw: real_dit(**{**kw, "dim": 32, "depth": 2, "heads": 2, "dim_head": 16, "text_dim": BRIDGE})

calls = {"n": 0}
real_randn = torch.randn


def noise(shape):
    """[B, C, T]: sin(1.3·t + 0.7·c + 0.11·call)."""
    b, c, t = shape
    call = calls["n"]
    calls["n"] += 1
    ci = torch.arange(c, dtype=torch.float64).view(1, c, 1)
    ti = torch.arange(t, dtype=torch.float64).view(1, 1, t)
    return torch.sin(1.3 * ti + 0.7 * ci + 0.11 * call).expand(b, c, t).clone()


def patched_randn(*args, **kwargs):
    shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)) else args
    return noise(list(shape))


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def build(version):
    model = models.SynthesizerTrnV3(
        spec_channels=1025, segment_size=20480 // 640, inter_channels=8, hidden_channels=HIDDEN, filter_channels=32,
        n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, resblock="1", resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]], upsample_rates=[4], upsample_initial_channel=32, upsample_kernel_sizes=[8],
        gin_channels=GIN, semantic_frame_rate="25hz", freeze_quantizer=True, version=version)
    model.ssl_proj = nn.Conv1d(SSL, SSL, 2, stride=2)
    model.quantizer = ResidualVectorQuantizer(dimension=SSL, n_q=1, bins=64)
    model.enc_p.ssl_proj = nn.Conv1d(SSL, HIDDEN, 1)
    model.enc_p.mrte = MRTE(content_enc_channels=HIDDEN, hidden_size=GIN, out_channels=HIDDEN, n_heads=4)
    model.ref_enc = MelStyleEncoder(704, style_hidden=16, style_vector_dim=GIN)
    model.bridge = nn.Sequential(nn.Conv1d(HIDDEN, BRIDGE, 1), nn.LeakyReLU())
    model.wns1 = models.Encoder(BRIDGE, BRIDGE, BRIDGE, 5, 1, 8, gin_channels=GIN)
    model.linear_mel = nn.Conv1d(BRIDGE, 100, 1)
    model = model.double().eval()
    with torch.no_grad():
        cb = model.quantizer.vq.layers[0]._codebook
        cb.embed.copy_(torch.randn(64, SSL))
        cb.embed_avg.copy_(cb.embed)
        cb.inited.fill_(1)
        # adaLN-zero, GRN and the output projection start at zero in F5-TTS; give them values so the fixture tests them.
        for name, p in model.named_parameters():
            if name.startswith("cfm.") and ("attn_norm.linear" in name or "norm_out.linear" in name or "proj_out" in name
                                            or "time_embed" in name or "d_embed" in name or ".grn." in name):
                p.copy_(torch.randn_like(p) * 0.2)
    return model


def chunked(model, version, fea_ref, fea_todo, mel2, steps, cfg):
    """inference_webui.get_tts_wav's loop (v3/v4) or synthesize_v5_mel."""
    if version in models_v5.V5_VERSIONS:
        return synthesize_v5_mel(model, fea_ref, fea_todo, mel2, steps, cfg)
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :T_min]
    fea_ref = fea_ref[:, :, :T_min]
    Tref = 468 if version == "v3" else 500
    Tchunk = 934 if version == "v3" else 1000
    if T_min > Tref:
        mel2 = mel2[:, :, -Tref:]
        fea_ref = fea_ref[:, :, -Tref:]
        T_min = Tref
    chunk_len = Tchunk - T_min
    results = []
    idx = 0
    while 1:
        fea_todo_chunk = fea_todo[:, :, idx: idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break
        idx += chunk_len
        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
        cfm_res = model.cfm.inference(fea, torch.LongTensor([fea.size(1)]), mel2, steps, inference_cfg_rate=0)
        cfm_res = cfm_res[:, :, mel2.shape[2]:]
        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]
        results.append(cfm_res)
    return torch.cat(results, 2)


def loss(model, ssl, spec, mel, text, prompt_u, t, bootstrap, k):
    """forward with its draws fixed: torch.rand gives the prompt fraction, then t; random.random the bootstrap coin;
    torch.randint the shortcut exponent; torch.randn_like the noise pattern."""
    rands = iter([torch.tensor([prompt_u]), torch.tensor([t])])
    saved = (torch.rand, torch.randn_like, torch.randint, random.random)
    torch.rand = lambda *a, **kw: next(rands)
    torch.randn_like = lambda x, **kw: noise(list(x.shape))
    torch.randint = lambda *a, **kw: torch.tensor([k])
    random.random = lambda: 0.1 if bootstrap else 0.9
    try:
        with torch.no_grad():
            value = model(ssl, spec, mel, None, torch.LongTensor([spec.shape[2]]), text, torch.LongTensor([text.shape[1]]),
                          torch.LongTensor([mel.shape[2]]), False)
    finally:
        torch.rand, torch.randn_like, torch.randint, random.random = saved
    return value.item()


def lora(model):
    """s2_train_v3_lora.py's adapters (rank 4 here, alpha = rank) with lora_B given values, the inference of the adapted
    CFM, and of the merged one (merge_and_unload)."""
    from peft import LoraConfig, get_peft_model
    model.cfm = get_peft_model(model.cfm, LoraConfig(target_modules=["to_k", "to_q", "to_v", "to_out.0"], r=4, lora_alpha=4,
                                                     init_lora_weights=True))
    with torch.no_grad():
        for name, p in model.cfm.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn_like(p) * 0.1)
    state = {k.replace("base_model.model.", ""): tolist(v) for k, v in model.cfm.state_dict().items() if "lora_" in k}
    mu = real_randn(1, 9, BRIDGE)
    prompt = real_randn(1, 100, 3)
    torch.randn = patched_randn
    with torch.no_grad():
        calls["n"] = 0
        adapted = model.cfm.inference(mu, torch.LongTensor([9]), prompt, 3, inference_cfg_rate=0)
        merged_cfm = model.cfm.merge_and_unload()
        calls["n"] = 0
        merged = merged_cfm.inference(mu, torch.LongTensor([9]), prompt, 3, inference_cfg_rate=0)
    torch.randn = real_randn
    return {"rank": 4, "state": state, "mu": tolist(mu), "prompt": tolist(prompt), "adapted": tolist(adapted), "merged": tolist(merged)}


def main():
    torch.manual_seed(7)
    torch.set_default_dtype(torch.float64)
    out = {"ssl_dim": SSL, "hidden": HIDDEN, "gin": GIN, "bridge": BRIDGE, "stride": STRIDE, "versions": {}}
    for version in ["v3", "v4", "v5dev"]:
        model = build(version)
        entry = {"symbols": model.enc_p.text_embedding.num_embeddings,
                 "state": {k: tolist(v) for k, v in model.state_dict().items()}}
        with torch.no_grad():
            reference = torch.randn(1, 1025, 11).abs()
            ge = model.ref_enc(reference[:, :704] * torch.ones(1, 1, 11), torch.ones(1, 1, 11))
            entry.update(reference=tolist(reference), ge=tolist(ge))
            codes = torch.randint(0, 64, (1, 1, 6))
            text = torch.randint(0, entry["symbols"], (1, 5))
            entry.update(codes=codes[0, 0].tolist(), text=text[0].tolist())
            decodes = []
            for speed in [1.0, 1.25]:
                fea, _ = model.decode_encp(codes, text, None, ge=ge, speed=speed)
                decodes.append({"speed": speed, "shape": list(fea.shape), "fea": tolist(fea)})
            entry["decodes"] = decodes
            torch.randn = patched_randn
            # One CFM call: 9 feature frames, a 3-frame prompt, 3 steps (V5 with its default CFG).
            calls["n"] = 0
            mu = real_randn(1, 9, BRIDGE)
            prompt = real_randn(1, 100, 3)
            cfg = 1.3 if version == "v5dev" else 0.0
            result = model.cfm.inference(mu, torch.LongTensor([9]), prompt, 3, inference_cfg_rate=cfg)
            entry["inference"] = {"mu": tolist(mu), "prompt": tolist(prompt), "steps": 3, "cfg": cfg, "mel": tolist(result)}
            # Chunked synthesis: a 30-frame reference and a target past one chunk.
            ref_frames = 30
            target = {"v3": 950, "v4": 1010, "v5dev": 700}[version]
            fea_ref = real_randn(1, BRIDGE, ref_frames + 2) * 0.5
            mel2 = real_randn(1, 100, ref_frames) * 0.5
            fea_todo = torch.sin(torch.arange(target, dtype=torch.float64).view(1, 1, -1) * 0.05
                                 + torch.arange(BRIDGE, dtype=torch.float64).view(1, -1, 1) * 0.3)
            calls["n"] = 0
            mel = chunked(model, version, fea_ref, fea_todo, mel2, 2, cfg)
            torch.randn = real_randn
            entry["chunked"] = {"fea_ref": tolist(fea_ref), "mel2": tolist(mel2), "target": target, "steps": 2, "cfg": cfg,
                                "shape": list(mel.shape), "calls": calls["n"], "mel": mel.reshape(-1)[::STRIDE].tolist(),
                                "sum": mel.sum().item()}
        ssl = real_randn(1, SSL, 8)
        spec = real_randn(1, 1025, 8).abs()
        mel_frames = 15 if version == "v3" else 16
        mel = real_randn(1, 100, mel_frames) * 0.5
        losses = []
        for bootstrap, k in ([(False, 3)] if version in models_v5.V5_VERSIONS else [(False, 3), (True, 3), (True, 7)]):
            calls["n"] = 0
            value = loss(model, ssl, spec, mel, text, 0.6, 0.35, bootstrap, k)
            losses.append({"bootstrap": bootstrap, "k": k, "prompt_u": 0.6, "t": 0.35, "loss": value})
        entry["loss"] = {"ssl": tolist(ssl), "spec": tolist(spec), "mel": tolist(mel), "mel_frames": mel_frames, "cases": losses}
        if version == "v3":
            entry["lora"] = lora(model)
        out["versions"][version] = entry
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "gptsovits_cfm_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
