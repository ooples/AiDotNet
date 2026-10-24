"""CosyVoice 2 flow reference: FunAudioLLM/CosyVoice's CausalMaskedDiffWithXvec (cosyvoice/flow/flow.py) with its
UpsampleConformerEncoder, CausalConditionalCFM and CausalConditionalDecoder. Pass checkouts with --cosyvoice and --matcha.

Builds a tiny model whose weights follow an exact integer formula of each parameter's name and index (weights(), which
the C# test repeats, so no weights are stored; the encoder is 512 wide and the mel 80 bands because the reference
hard-codes both) and runs it in float64 with dropout 0, recording in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice2_flow_reference.json, offline and streaming (chunks of 2
tokens, 4 frames): the projected x-vector, the encoder's states, the flow-matching loss of a target with a prompt
condition at a given t and noise, with and without condition dropout; the sample of Euler steps from the fixed noise
for a prompt and a text (CausalMaskedDiffWithXvec.inference, which returns float32: a float64 Euler loop that rounds to
it exactly is recorded); and the offline loss after one SGD step (lr 1e-3). Default: verify; --write: regenerate.
"""
import argparse
import math

import torch

import _fixture
from cosyvoice_flow_reference import stub_imports

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice2_flow_reference.json"
MEL = 80
CONFIG = dict(input_size=512, output_size=MEL, spk_embed_dim=6, vocab_size=20, input_frame_rate=25, token_mel_ratio=2,
              pre_lookahead_len=3)
ENCODER = dict(output_size=512, attention_heads=8, linear_units=32, num_blocks=1, static_chunk_size=2)
DECODER = dict(channels=[16], attention_head_dim=8, n_blocks=1, num_mid_blocks=1, num_heads=2, static_chunk_size=4)
TOKENS = [3, 17, 8, 1, 11, 6, 14]
PROMPT = [9, 2, 15]
T = 0.37
STEPS = 3
LEARNING_RATE = 1e-3


def weights(name, count):
    h = sum((j + 1) * ord(ch) for j, ch in enumerate(name)) % 100003
    base = 1.0 if ("norm" in name and name.endswith("weight")) else 0.0
    return [base + 0.1 * (((i * 7919 + h) % 2003) / 2003.0 - 0.5) for i in range(count)]


def build():
    from omegaconf import DictConfig
    from cosyvoice.flow.decoder import CausalConditionalDecoder
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.transformer.upsample_encoder import UpsampleConformerEncoder
    torch.set_default_dtype(torch.float64)
    encoder = UpsampleConformerEncoder(input_size=512, dropout_rate=0.0, positional_dropout_rate=0.0, attention_dropout_rate=0.0,
                                       normalize_before=True, input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                                       selfattention_layer_type="rel_selfattn", use_cnn_module=False, macaron_style=False, **ENCODER)
    estimator = CausalConditionalDecoder(in_channels=4 * MEL, out_channels=MEL, dropout=0.0, act_fn="gelu",
                                         num_decoding_left_chunks=-1, **DECODER)
    cfm = CausalConditionalCFM(in_channels=3 * MEL, n_spks=1, spk_emb_dim=MEL, estimator=estimator,
                               cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                                                      "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}))
    flow = CausalMaskedDiffWithXvec(output_type="mel", only_mask_loss=True, encoder=encoder, decoder=cfm, **CONFIG).eval()
    with torch.no_grad():
        for name, p in flow.named_parameters():
            p.copy_(torch.tensor(weights(name, p.numel())).reshape(p.shape))
    return flow


def pieces(flow, xvector, tokens, streaming):
    embedding = flow.spk_embed_affine_layer(torch.nn.functional.normalize(xvector, dim=1))
    token = flow.input_embedding(torch.clamp(torch.tensor([tokens]), min=0))
    h, _ = flow.encoder(token, torch.tensor([len(tokens)]), streaming=streaming)
    return embedding, flow.encoder_proj(h).transpose(1, 2)


def loss(flow, target, mu, spks, cond, noise, drop, streaming):
    # compute_loss with t, z and the condition dropout given.
    sigma_min = flow.decoder.sigma_min
    t = torch.tensor([[[T]]])
    y = (1 - (1 - sigma_min) * t) * noise + t * target
    u = target - (1 - sigma_min) * noise
    if drop:
        mu, spks, cond = mu * 0, spks * 0, cond * 0
    mask = torch.ones(1, 1, target.shape[2])
    pred = flow.decoder.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
    return torch.nn.functional.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])


def sample(flow, xvector, prompt_mel, streaming):
    # CausalMaskedDiffWithXvec.inference, and the same Euler loop in float64.
    with torch.no_grad():
        frames = 2 * (len(PROMPT) + len(TOKENS))
        reference, _ = flow.inference(torch.tensor([TOKENS]), torch.tensor([len(TOKENS)]), torch.tensor([PROMPT]),
                                      torch.tensor([len(PROMPT)]), prompt_mel, torch.tensor([prompt_mel.shape[1]]), xvector,
                                      streaming=streaming, finalize=True)
        spks, mu = pieces(flow, xvector, PROMPT + TOKENS, streaming)
        cond = torch.zeros(1, MEL, frames)
        cond[:, :, :prompt_mel.shape[1]] = prompt_mel.transpose(1, 2)
        noise = flow.decoder.rand_noise[:, :, :frames].to(torch.float64)
        t_span = 1 - torch.cos(torch.linspace(0, 1, 11) * 0.5 * torch.pi)
        mask = torch.ones(1, 1, frames)
        x, t, dt = noise.clone(), t_span[0], t_span[1] - t_span[0]
        w = flow.decoder.inference_cfg_rate
        for step in range(1, len(t_span)):
            guided = flow.decoder.estimator(x, mask, mu, t.unsqueeze(0), spks, cond, streaming=streaming)
            free = flow.decoder.estimator(x, mask, mu * 0, t.unsqueeze(0), spks * 0, cond * 0, streaming=streaming)
            x = x + dt * ((1.0 + w) * guided - w * free)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        generated = x[:, :, prompt_mel.shape[1]:]
        assert torch.equal(generated.float(), reference), "the float64 Euler loop does not round to the reference's sample"
        return noise, generated


def run(flow, xvector, target, cond, noise, prompt_mel):
    out = {}
    for streaming in (False, True):
        key = "streaming" if streaming else "offline"
        spks, mu = pieces(flow, xvector, TOKENS, streaming)
        out[key] = {"speaker": spks, "mu": mu,
                    "loss": loss(flow, target, mu, spks, cond, noise, False, streaming),
                    "loss_dropped": loss(flow, target, mu, spks, cond, noise, True, streaming)}
        out[key]["sample_noise"], out[key]["sample"] = sample(flow, xvector, prompt_mel, streaming)
    plain = out["offline"]["loss"]
    parameters = list(flow.parameters())
    grads = torch.autograd.grad(plain, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
    spks, mu = pieces(flow, xvector, TOKENS, False)
    out["loss_after_sgd"] = loss(flow, target, mu, spks, cond, noise, False, False)
    return out


def flatten(out):
    result = {}
    for key, value in out.items():
        if isinstance(value, dict):
            result[key] = {k: [float(x) for x in v.detach().reshape(-1).tolist()] for k, v in value.items()}
        else:
            result[key] = float(value.detach())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stub_imports(args.cosyvoice, args.matcha)
    frames = 2 * len(TOKENS)
    if args.write:
        flow = build()
        generator = torch.Generator().manual_seed(7)
        tensors = {"xvector": torch.randn(1, CONFIG["spk_embed_dim"], generator=generator),
                   "target": torch.randn(1, MEL, frames, generator=generator),
                   "cond": torch.randn(1, MEL, frames, generator=generator),
                   "noise": torch.randn(1, MEL, frames, generator=generator),
                   "prompt_mel": torch.randn(1, 2 * len(PROMPT), MEL, generator=generator)}
        tensors["cond"][:, :, frames // 2:] = 0
        data = {"config": CONFIG, "encoder": ENCODER, "decoder": DECODER, "tokens": TOKENS, "prompt": PROMPT, "t": T,
                "learning_rate": LEARNING_RATE, "parameters": [[n, list(p.shape)] for n, p in flow.named_parameters()]}
        for k, v in tensors.items():
            data[k] = [float(x) for x in v.reshape(-1).tolist()]
        data.update(flatten(run(flow, **tensors)))
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    flow = build()
    shapes = {"xvector": (1, CONFIG["spk_embed_dim"]), "target": (1, MEL, frames), "cond": (1, MEL, frames),
              "noise": (1, MEL, frames), "prompt_mel": (1, 2 * len(PROMPT), MEL)}
    tensors = {k: torch.tensor(data[k]).reshape(s) for k, s in shapes.items()}
    out = flatten(run(flow, **tensors))
    check = _fixture.Comparison("CosyVoice 2 flow")
    for mode in ("offline", "streaming"):
        for k, v in out[mode].items():
            check.floats(f"{mode} {k}", data[mode][k], v, 1e-12, float)
    check.floats("loss_after_sgd", [data["loss_after_sgd"]], [out["loss_after_sgd"]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
