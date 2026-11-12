"""Reference fixture for MeloTTS: the official myshell-ai/MeloTTS SynthesizerTrn (Bert-VITS2 lineage) in float64.

MeloTTS has no paper; its code is the reference. At small sizes, one utterance through SynthesizerTrn.forward with every
random draw fixed (the posterior noise, the alignment-search noise, the stochastic duration predictor's posterior noise
and the decoder's segment): the alignment, the prior and posterior statistics, the flow output, both duration losses,
the duration discriminator's outputs and the decoder's waveform; then a combined loss (both duration losses, the KL of
train.py's kl_loss, the duration discriminator's generator term and a fixed weighting of the waveform) and one SGD step.
Generation (SynthesizerTrn.infer) with fixed noise follows, at an sdp_ratio of 0.2.

Needs a clone of myshell-ai/MeloTTS at MELO_COMMIT next to this repository (../refsrc/MeloTTS). MeloTTS's alignment
search runs in float32 under numba; here the same algorithm runs in float64 so the comparison measures the network.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_reference.json. Run with the reference venv.
"""
import json
import math
import os
import sys
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc"))
MELO_COMMIT = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
sys.path.insert(0, os.path.join(REFSRC, "MeloTTS"))
torch.set_default_dtype(torch.float64)

from melo import commons, models, monotonic_align  # noqa: E402
from melo import losses as melo_losses  # noqa: E402

# melo.losses casts every input to float32 (for mixed precision); the same formulas in float64.
_float = torch.Tensor.float


def _in_float64(fn):
    def run(*args, **kwargs):
        with mock.patch.object(torch.Tensor, "float", lambda t: t.double()):
            return fn(*args, **kwargs)
    return run


kl_loss = _in_float64(melo_losses.kl_loss)
generator_loss = _in_float64(melo_losses.generator_loss)


def maximum_path_float64(neg_cent, mask):
    # melo.monotonic_align.core.maximum_path_jit, in float64.
    values = neg_cent.detach().cpu().numpy().astype(np.float64).copy()
    paths = np.zeros(values.shape, dtype=np.int32)
    t_ys = mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
    t_xs = mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
    for i in range(values.shape[0]):
        path, value, t_y, t_x = paths[i], values[i], t_ys[i], t_xs[i]
        index = t_x - 1
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur = -1e9 if x == y else value[y - 1, x]
                v_prev = (0.0 if y == 0 else -1e9) if x == 0 else value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index -= 1
    return torch.from_numpy(paths).to(dtype=neg_cent.dtype)


models.monotonic_align.maximum_path = maximum_path_float64

CONFIG = dict(vocab=12, spec=9, segment=8, inter=4, hidden=8, filter=12, heads=2, layers=3, kernel=3, kernels=[3],
              dilations=[[1, 3, 5]], rates=[2, 2], initial=8, up_kernels=[4, 4], speakers=3, gin=6, flows=2, flow_layers=3,
              languages=2, tones=3, tokens=5, frames=14, sid=1, mas_noise=0.01)


def flat(module):
    return {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}


def main():
    c = CONFIG
    gen = torch.Generator().manual_seed(7)
    net = models.SynthesizerTrn(
        c["vocab"], c["spec"], c["segment"], c["inter"], c["hidden"], c["filter"], c["heads"], c["layers"], c["kernel"], 0.0,
        "1", c["kernels"], c["dilations"], c["rates"], c["initial"], c["up_kernels"], n_speakers=c["speakers"],
        gin_channels=c["gin"], use_sdp=True, n_flow_layer=c["flows"], n_layers_trans_flow=c["flow_layers"],
        flow_share_parameter=False, use_transformer_flow=True, num_languages=c["languages"], num_tones=c["tones"],
        use_spk_conditioned_encoder=True, use_noise_scaled_mas=True, mas_noise_scale_initial=c["mas_noise"])
    disc = models.DurationDiscriminator(c["hidden"], c["filter"], 3, 0.0, gin_channels=c["gin"])
    with torch.no_grad():
        for p in list(net.parameters()) + list(disc.parameters()):
            p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    initial, disc_state = flat(net), flat(disc)

    L, T = c["tokens"], c["frames"]
    x = torch.randint(0, c["vocab"], (1, L), generator=gen)
    tone = torch.randint(0, c["tones"], (1, L), generator=gen)
    language = torch.randint(0, c["languages"], (1, L), generator=gen)
    bert = torch.randn(1, 1024, L, generator=gen) * 0.1
    ja_bert = torch.randn(1, 768, L, generator=gen) * 0.1
    spec = torch.randn(1, c["spec"], T, generator=gen).abs()
    sid = torch.tensor([c["sid"]])
    posterior_noise = torch.randn(1, c["inter"], T, generator=gen)
    mas_noise = torch.randn(1, T, L, generator=gen)
    sdp_noise = torch.randn(1, 2, L, generator=gen)
    slice_start = 3
    weights = torch.randn(1, 1, c["segment"] * 2 * 2, generator=gen)
    net.current_mas_noise_scale = c["mas_noise"]

    randn_like = iter([posterior_noise, mas_noise])
    with mock.patch("torch.randn_like", side_effect=lambda *a, **k: next(randn_like).clone()), \
            mock.patch("torch.randn", side_effect=lambda *a, **k: sdp_noise.clone()), \
            mock.patch("torch.rand", side_effect=lambda *a, **k: torch.tensor([slice_start / (T - c["segment"] + 1)])):
        net.train()
        # The duration predictors hard-code dropout 0.5 (models.SynthesizerTrn); evaluate them without it.
        net.sdp.eval()
        net.dp.eval()
        disc.train()
        o, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (hidden, logw, logw_) = net(
            x, torch.tensor([L]), spec, torch.tensor([T]), sid, tone, language, bert, ja_bert)
    assert int(ids_slice[0]) == slice_start, ids_slice
    g = net.emb_g(sid).unsqueeze(-1)
    # As train.py calls it: no speaker, the predicted log-durations as dur_r and the alignment's as dur_hat.
    y_dur_hat_r, y_dur_hat_g = disc(hidden, x_mask, logw, logw_)
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
    loss_dur_gen, _ = generator_loss(y_dur_hat_g)
    loss = l_length.sum() + loss_kl + loss_dur_gen + (o * weights).sum()

    # The SDP's and DP's losses separately, recomputed from the same draws.
    params = list(net.parameters())
    optimizer = torch.optim.SGD(params, lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    out = {
        "melo_commit": MELO_COMMIT, "config": CONFIG, "initial": initial, "duration_discriminator": disc_state,
        "x": x.flatten().tolist(), "tone": tone.flatten().tolist(), "language": language.flatten().tolist(),
        "bert": bert.flatten().tolist(), "ja_bert": ja_bert.flatten().tolist(), "spec": spec.flatten().tolist(),
        "posterior_noise": posterior_noise.flatten().tolist(), "mas_noise": mas_noise.flatten().tolist(),
        "sdp_noise": sdp_noise.flatten().tolist(), "slice_start": slice_start, "weights": weights.flatten().tolist(),
        "attn": attn[0, 0].flatten().tolist(), "m_p": m_p.flatten().tolist(), "logs_p": logs_p.flatten().tolist(),
        "m_q": m_q.flatten().tolist(), "logs_q": logs_q.flatten().tolist(), "z": z.flatten().tolist(),
        "z_p": z_p.flatten().tolist(), "logw": logw.flatten().tolist(), "logw_": logw_.flatten().tolist(),
        "l_length": float(l_length.sum()), "loss_kl": float(loss_kl), "dur_real": y_dur_hat_r.flatten().tolist(),
        "dur_generated": y_dur_hat_g.flatten().tolist(), "o": o.flatten().tolist(), "loss": float(loss), "lr": 0.01,
        "final": flat(net),
    }

    # Generation with the updated network, fixed noise for the SDP and the prior, sdp_ratio 0.2.
    net.eval()
    infer_sdp = torch.randn(1, 2, L, generator=gen)
    real_randn = torch.randn
    with mock.patch("torch.randn", side_effect=lambda *a, **k: infer_sdp.clone()), \
            torch.no_grad():
        # infer draws z_p's noise with randn_like after the SDP's randn; record it from a fixed generator.
        prior_holder = {}

        def prior_noise(t, *a, **k):
            prior_holder["noise"] = real_randn(t.shape, generator=gen)
            return prior_holder["noise"].clone()

        with mock.patch("torch.randn_like", side_effect=prior_noise):
            audio, attn_i, y_mask_i, (z_i, z_p_i, m_p_i, logs_p_i) = net.infer(
                x, torch.tensor([L]), sid, tone, language, bert, ja_bert, noise_scale=0.667, length_scale=1.0,
                noise_scale_w=0.8, sdp_ratio=0.2)
    out["infer"] = {"sdp_noise": infer_sdp.flatten().tolist(), "prior_noise": prior_holder["noise"].flatten().tolist(),
                    "frames": int(y_mask_i.sum()), "attn": attn_i[0, 0].flatten().tolist(), "audio": audio.flatten().tolist()}

    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "melo_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), "loss", out["loss"], "l_length", out["l_length"], "kl", out["loss_kl"],
          "infer frames", out["infer"]["frames"])


if __name__ == "__main__":
    main()
