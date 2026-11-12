"""Reference fixture for GPT-SoVITS's SoVITS synthesizer (module/models.py SynthesizerTrn) for v1, v2 and v2ProPlus.

Runs the release's own module (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image in float64 with small sizes:
the widths the release fixes (SSL 768, MRTE 512, the 20480-dim speaker embedding) are replaced after construction by
modules of the same kind with smaller widths (ssl_proj, the quantizer, enc_p.ssl_proj, MRTE, ref_enc, sv_emb,
ge_to512).
Records per version: the state dict; extract_latent of random SSL features; the style vector of a reference
spectrogram (and speaker embedding); the text encoder's prior of quantized codes; the flow forward and reverse; the
generator; decode with the prior's noise fixed (at speed 1 and 1.25); and the posterior with its noise fixed.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_sovits_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_sovits_reference.json.
"""
import json
import os
import sys

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ.setdefault("version", "v2")

from module import models  # noqa: E402
from module.mrte_model import MRTE  # noqa: E402
from module.quantize import ResidualVectorQuantizer  # noqa: E402

SSL = 24
HIDDEN = 16


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def build(version):
    gin = 16
    mrte_hidden = gin if version not in models.v2pro_set else 24
    model = models.SynthesizerTrn(
        spec_channels=1025, segment_size=20480 // 640, inter_channels=8, hidden_channels=HIDDEN, filter_channels=32,
        n_heads=2, n_layers=2, kernel_size=3, p_dropout=0.1, resblock="1", resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], upsample_rates=[4, 2], upsample_initial_channel=32,
        upsample_kernel_sizes=[8, 4], gin_channels=gin, semantic_frame_rate="25hz", freeze_quantizer=True, version=version)
    model.ssl_proj = nn.Conv1d(SSL, SSL, 2, stride=2)
    model.quantizer = ResidualVectorQuantizer(dimension=SSL, n_q=1, bins=64)
    model.enc_p.ssl_proj = nn.Conv1d(SSL, HIDDEN, 1)
    model.enc_p.mrte = MRTE(content_enc_channels=HIDDEN, hidden_size=mrte_hidden, out_channels=HIDDEN, n_heads=4)
    from module.modules import MelStyleEncoder
    model.ref_enc = MelStyleEncoder(1025 if version == "v1" else 704, style_hidden=16, style_vector_dim=gin)
    if version in models.v2pro_set:
        model.sv_emb = nn.Linear(32, gin)
        model.ge_to512 = nn.Linear(gin, mrte_hidden)
    model = model.double().eval()
    with torch.no_grad():
        cb = model.quantizer.vq.layers[0]._codebook
        cb.embed.copy_(torch.randn(64, SSL))
        cb.embed_avg.copy_(cb.embed)
        cb.inited.fill_(1)
        for name, p in model.named_parameters():
            if "flow" in name and name.endswith("post.weight"):
                p.copy_(torch.randn_like(p) * 0.1)
            if "flow" in name and name.endswith("post.bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            if name == "prelu.weight":
                p.copy_(torch.rand_like(p))
    return model, gin, mrte_hidden


def main():
    torch.manual_seed(5)
    torch.set_default_dtype(torch.float64)
    out = {"ssl_dim": SSL, "hidden": HIDDEN, "versions": {}}
    for version in ["v1", "v2", "v2ProPlus"]:
        model, gin, mrte_hidden = build(version)
        state = {k: tolist(v) for k, v in model.state_dict().items()}
        entry = {"gin": gin, "mrte_hidden": mrte_hidden, "symbols": model.enc_p.text_embedding.num_embeddings, "state": state}
        with torch.no_grad():
            ssl = torch.randn(1, SSL, 14)
            entry["ssl"] = tolist(ssl)
            entry["codes"] = model.extract_latent(ssl)[0, 0].tolist()
            reference = torch.randn(1, 1025, 11).abs()
            sv = torch.randn(1, 32) if version in models.v2pro_set else None
            entry["reference"] = tolist(reference)
            entry["sv"] = None if sv is None else tolist(sv)
            refer_mask = torch.ones(1, 1, 11)
            ge = model.ref_enc(reference * refer_mask if version == "v1" else reference[:, :704] * refer_mask, refer_mask)
            if version in models.v2pro_set:
                ge = model.prelu(ge + model.sv_emb(sv).unsqueeze(-1))
            entry["style"] = tolist(ge)
            codes = torch.tensor([[[3, 17, 60, 2, 9, 41]]])
            text = torch.tensor([[5, 12, 7, 30, 2, 19, 4]])
            entry["decode_codes"] = codes[0, 0].tolist()
            entry["text"] = text[0].tolist()
            quantized = model.quantizer.decode(codes)
            quantized = nn.functional.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            mrte_ge = model.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if version in models.v2pro_set else ge
            y_lengths = torch.LongTensor([12])
            _, m_p, logs_p, y_mask, _, _ = model.enc_p(quantized, y_lengths, text, torch.LongTensor([7]), mrte_ge)
            entry["prior_mean"] = tolist(m_p)
            entry["prior_logs"] = tolist(logs_p)
            z = torch.randn(1, 8, 12)
            entry["flow_input"] = tolist(z)
            entry["flow_forward"] = tolist(model.flow(z, y_mask, g=ge))
            entry["flow_reverse"] = tolist(model.flow(z, y_mask, g=ge, reverse=True))
            entry["generator"] = tolist(model.dec(z, g=ge))
            decodes = []
            for speed in [1.0, 1.25]:
                noise_holder = {}
                original = torch.randn_like

                def fixed(t):
                    noise_holder["noise"] = original(t)
                    return noise_holder["noise"]

                torch.randn_like = fixed
                audio = model.decode(codes, text, reference, noise_scale=0.6, speed=speed, sv_emb=sv)
                torch.randn_like = original
                decodes.append({"speed": speed, "noise": tolist(noise_holder["noise"]), "noise_shape": list(noise_holder["noise"].shape),
                                "audio": tolist(audio)})
            entry["decodes"] = decodes
            spec = torch.randn(1, 1025, 9).abs()
            noise = torch.randn(1, 8, 9)
            original = torch.randn_like
            torch.randn_like = lambda t: noise
            zq, m_q, logs_q, _ = model.enc_q(spec, torch.LongTensor([9]), g=ge)
            torch.randn_like = original
            entry["posterior"] = {"spec": tolist(spec), "noise": tolist(noise), "z": tolist(zq), "mean": tolist(m_q), "logs": tolist(logs_q)}
        out["versions"][version] = entry
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "gptsovits_sovits_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
