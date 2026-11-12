"""Reference fixture for NaturalSpeech 2's modules from Amphion (open-mmlab/Amphion, models/tts/naturalspeech2), the
implementation that fills the paper's gaps, at a tiny configuration in float64: the prior encoder (phoneme encoder with
style-adaptive LayerNorm, duration and pitch predictors with prompt attention, length regulation and pitch embedding),
the prompt encoder and its query attention, the WaveNet denoiser, the diffusion's training terms with its draws fixed,
the RVQ cross-entropy distances, and Euler/midpoint reverse diffusion; and the flow-matching variant.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/Amphion-full gptsovits-ref \
        sh -c "pip install -q --no-deps encodec==0.1.1 && python /repo/tools/reference-data/naturalspeech2_reference.py"

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/naturalspeech2_reference.json.
"""
import json
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
# encodec imports torchaudio for file I/O only; the image has none.
sys.modules.setdefault("torchaudio", SimpleNamespace())

import encodec  # noqa: E402
from encodec.quantization import ResidualVectorQuantizer  # noqa: E402

_orig_24khz = encodec.EncodecModel.encodec_model_24khz
encodec.EncodecModel.encodec_model_24khz = staticmethod(lambda: _orig_24khz(pretrained=False))

from models.tts.naturalspeech2.ns2 import NaturalSpeech2  # noqa: E402
from models.tts.naturalspeech2.diffusion_flow import DiffusionFlow  # noqa: E402
from models.tts.naturalspeech2.ns2_loss import diff_loss, diff_ce_loss, log_dur_loss, log_pitch_loss  # noqa: E402


def ns(d):
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


LATENT, HIDDEN, NQ, BINS = 8, 16, 4, 16

CFG = {
    "latent_dim": LATENT,
    "prior_encoder": {
        "vocab_size": 20, "pitch_min": 50, "pitch_max": 1100, "pitch_bins_num": 16,
        "encoder": {"encoder_layer": 2, "encoder_hidden": HIDDEN, "encoder_head": 2, "conv_filter_size": 32,
                    "conv_kernel_size": 3, "encoder_dropout": 0.2, "use_cln": True},
        "duration_predictor": {"input_size": HIDDEN, "filter_size": HIDDEN, "kernel_size": 3, "conv_layers": 4,
                               "cross_attn_per_layer": 3, "attn_head": 2, "drop_out": 0.5},
        "pitch_predictor": {"input_size": HIDDEN, "filter_size": HIDDEN, "kernel_size": 5, "conv_layers": 4,
                            "cross_attn_per_layer": 3, "attn_head": 2, "drop_out": 0.5},
    },
    "diffusion": {
        "wavenet": {"input_size": LATENT, "hidden_size": HIDDEN, "out_size": LATENT, "num_layers": 5,
                    "cross_attn_per_layer": 3, "dilation_cycle": 2, "attn_head": 2, "drop_out": 0.2},
        "beta_min": 0.05, "beta_max": 20, "sigma": 1.0, "noise_factor": 1.0, "ode_solver": "euler",
        "diffusion_type": "diffusion",
    },
    "prompt_encoder": {"encoder_layer": 2, "encoder_hidden": HIDDEN, "encoder_head": 2, "conv_filter_size": 32,
                       "conv_kernel_size": 3, "encoder_dropout": 0.2, "use_cln": False},
    "query_emb": {"query_token_num": 4, "hidden_size": HIDDEN, "head_num": 2},
}


def randomize(module):
    # Random weights everywhere (the release zero-initializes the denoiser's output projection, which would hide it).
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.1) + (1.0 if name.endswith("ln.weight") else 0.0))


def main():
    torch.manual_seed(7)
    torch.set_default_dtype(torch.float64)
    model = NaturalSpeech2(ns(CFG)).double()
    model.quantizer = ResidualVectorQuantizer(dimension=LATENT, n_q=NQ, bins=BINS).double()
    randomize(model)
    with torch.no_grad():
        for layer in model.quantizer.vq.layers:
            layer._codebook.embed.copy_(torch.randn_like(layer._codebook.embed))
            layer._codebook.inited.fill_(1)
        # Durations of about two frames, so inference's rounded durations expand to a sequence the pitch convs can read.
        model.prior_encoder.duration_predictor.linear.weight.mul_(0.05)
        model.prior_encoder.duration_predictor.linear.bias.fill_(1.1)
    model.eval()

    n, frames, ref_frames = 6, 11, 5
    phone_id = torch.tensor([[3, 7, 12, 5, 9, 15]])
    duration = torch.tensor([[2, 1, 3, 0, 2, 3]])
    pitch = torch.tensor([[120.0, 130.5, 140.2, 155.0, 90.0, 210.0, 300.0, 49.0, 1200.0, 180.0, 175.0]])
    code = torch.randint(0, BINS, (1, NQ, frames))
    ref_code = torch.randint(0, BINS, (1, NQ, ref_frames))
    phone_mask = torch.ones(1, n)
    mask = torch.ones(1, frames)
    ref_mask = torch.ones(1, ref_frames)

    out = {"config": CFG, "phone_id": phone_id[0].tolist(), "duration": duration[0].tolist(), "pitch": pitch[0].tolist(),
           "code": code[0].tolist(), "ref_code": ref_code[0].tolist()}
    with torch.no_grad():
        ref_latent = model.code_to_latent(ref_code)
        latent = model.code_to_latent(code)
        ref_hidden = model.prompt_lin(ref_latent.transpose(1, 2))
        ref_hidden = model.prompt_encoder(ref_hidden, ref_mask, condition=None)
        spk_emb = ref_hidden.transpose(1, 2)
        query = model.query_emb(torch.arange(model.query_emb_num)).repeat(1, 1, 1)
        query, _ = model.query_attn(query, spk_emb.transpose(1, 2), spk_emb.transpose(1, 2), key_padding_mask=~(ref_mask.bool()))
        prior = model.prior_encoder(phone_id=phone_id, duration=duration, pitch=pitch, phone_mask=phone_mask, mask=mask,
                                    ref_emb=spk_emb, ref_mask=ref_mask, is_inference=False)
        cond = prior["prior_out"]

        # The diffusion's training terms at a fixed step and noise.
        t = torch.tensor([0.37])
        z = torch.randn(latent.shape)
        diff = model.diffusion
        cum_beta = diff.get_cum_beta(t.unsqueeze(-1).unsqueeze(-1))
        xt = latent * torch.exp(-0.5 * cum_beta) + z * torch.sqrt(1 - torch.exp(-cum_beta))
        x0_pred = diff.diff_estimator(xt, mask, cond, t, query)
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta)
        noise_pred = (xt - mean_pred) / torch.sqrt(1.0 - torch.exp(-cum_beta))
        pred_indices, pred_dist = model.latent_to_code(x0_pred, nq=NQ)
        gt_indices, _ = model.latent_to_code(latent, nq=NQ)
        out["latent"] = tolist(latent)
        out["ref_latent"] = tolist(ref_latent)
        out["prompt"] = tolist(ref_hidden)
        out["query"] = tolist(query)
        out["prior"] = {"prior_out": tolist(cond), "dur_pred_log": tolist(prior["dur_pred_log"]),
                        "pitch_pred_log": tolist(prior["pitch_pred_log"]), "pitch_token": prior["pitch_token"][0].tolist(),
                        "dur_loss": float(log_dur_loss(prior["dur_pred_log"], duration, phone_mask)),
                        "pitch_loss": float(log_pitch_loss(prior["pitch_pred_log"], pitch, mask))}
        out["diffusion"] = {"t": float(t[0]), "z": tolist(z), "xt": tolist(xt), "x0_pred": tolist(x0_pred),
                            "noise_pred": tolist(noise_pred), "pred_indices": tolist(pred_indices), "pred_dist": tolist(pred_dist),
                            "gt_indices": tolist(gt_indices),
                            "x0_loss": float(diff_loss(x0_pred, latent, mask)), "noise_loss": float(diff_loss(noise_pred, z, mask)),
                            "ce_loss": float(diff_ce_loss(pred_dist, gt_indices, mask))}

        # Reverse diffusion from a fixed start, both solvers.
        start = torch.randn(1, LATENT, frames)
        euler = diff.reverse_diffusion(start, None, cond, 3, query)
        diff.cfg.ode_solver = "midpoint"
        midpoint = diff.reverse_diffusion(start, None, cond, 3, query)
        diff.cfg.ode_solver = "euler"
        out["reverse"] = {"start": tolist(start), "steps": 3, "euler": tolist(euler), "midpoint": tolist(midpoint)}

        # Inference's prior: predicted durations (rounded) and pitch.
        inf = model.prior_encoder(phone_id=phone_id, duration=None, pitch=None, phone_mask=None, mask=None, ref_emb=spk_emb,
                                  ref_mask=ref_mask, is_inference=True)
        out["inference"] = {"dur_pred_round": inf["dur_pred_round"][0].tolist(), "prior_out": tolist(inf["prior_out"]),
                            "pitch_token": inf["pitch_token"][0].tolist()}

        # The flow-matching variant over the same denoiser weights.
        flow = DiffusionFlow(ns(CFG["diffusion"])).double()
        flow.diff_estimator.load_state_dict(diff.diff_estimator.state_dict())
        flow.eval()
        xt_flow = (1 - t) * latent + t * z
        flow_pred = flow.diff_estimator(xt_flow, mask, cond, t, query)
        flow_rev = flow.reverse_diffusion(start, None, cond, 3, query)
        out["flow"] = {"xt": tolist(xt_flow), "flow_pred": tolist(flow_pred),
                       "loss": float(diff_loss(flow_pred, z - latent, mask)), "reverse": tolist(flow_rev)}

    out["state"] = {k: tolist(v) for k, v in model.state_dict().items()
                    if (not k.startswith("quantizer.") or k.endswith("_codebook.embed")) and not k.endswith("position_emb.pe")}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "naturalspeech2_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
