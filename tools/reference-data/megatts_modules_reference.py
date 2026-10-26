"""Mega-TTS's building blocks through the code they come from, on tiny sizes.

Mega-TTS (Jiang et al. 2023) has no public code. Its blocks are the authors' NATSpeech modules (NATSpeech/NATSpeech: the
FastSpeech content encoder, the duration predictor, the convolution stacks, the decoder layers of the prosody language
model) and SyntaSpeech's multi-window discriminator (yerfor/SyntaSpeech), at the sizes of the paper's Table 7. This script
runs those modules on random weights in float64 and records, in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts_modules_reference.json, their outputs:
- FastSpeechEncoder (8 tokens with padding, width 16, two layers of two heads, kernel 5);
- DurationPredictor (three convolution layers of width 16, kernel 3);
- ConvBlocks (two residual blocks of two layers, kernel 5, layer norm, a kernel-3 post-net to 12 channels), on a padded
  sequence;
- two TransformerDecoderLayers (width 16, two heads, kernel-5 left-padded feed-forward) with the causal mask, as the
  prosody language model stacks them, after FFTBlocks' sinusoidal positions (a precision patch builds the positions in float64, as
  NATSpeech builds them in float32);
- the multi-window discriminator (windows 8 and 16, 20 mel bins, width 6) at given window starts, in evaluation mode.
Default: verify; --write: regenerate. Pass --natspeech and --syntaspeech (checkouts). Needs torch.
"""
import argparse
import os
import sys

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts_modules_reference.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natspeech", required=True)
    parser.add_argument("--syntaspeech", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.natspeech))
    from modules.commons.transformer import FastSpeechEncoder, TransformerDecoderLayer, SinusoidalPositionalEmbedding
    from modules.commons.nar_tts_modules import DurationPredictor
    from modules.commons.conv import ConvBlocks
    import math
    import modules.commons.transformer as natt

    # Precision patch: the sinusoidal positions are built in float32 (an explicit dtype); the same in float64.
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    natt.SinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding)
    import importlib.util
    spec = importlib.util.spec_from_file_location("mwd", os.path.join(args.syntaspeech, "modules", "tts", "syntaspeech", "multi_window_disc.py"))
    mwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mwd)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(223)
    encoder = FastSpeechEncoder(20, hidden_size=16, num_layers=2, kernel_size=5, num_heads=2)
    duration = DurationPredictor(16, n_layers=3, n_chans=16, kernel_size=3, dropout_rate=0.0)
    blocks = ConvBlocks(16, 12, None, 5, num_layers=2, layers_in_block=2, c_multiple=2, ln_eps=1e-5, post_net_kernel=3)
    decoder = torch.nn.ModuleList([TransformerDecoderLayer(16, 0.0, kernel_size=5, num_heads=2) for _ in range(2)])
    disc = mwd.MultiWindowDiscriminator([8, 16], freq_length=20, hidden_size=6)
    modules = {"encoder": encoder, "duration": duration, "blocks": blocks, "decoder": decoder, "disc": disc}
    with torch.no_grad():
        for prefix, module in modules.items():
            for name, p in module.named_parameters():
                key = f"{prefix}.{name}"
                if stored is not None:
                    p.copy_(torch.tensor(stored["state"][key], dtype=torch.float32).reshape(p.shape))
                elif name.endswith("weight") and p.dim() == 1:
                    p.copy_((1 + 0.1 * torch.randn_like(p)).float())
                else:
                    p.copy_((0.3 * torch.randn_like(p)).float())
            for name, b in module.named_buffers():
                if "running" in name and stored is None:
                    b.copy_(((torch.rand_like(b) + 0.5) if "var" in name else 0.1 * torch.randn_like(b)).float())
                elif "running" in name:
                    b.copy_(torch.tensor(stored["state"][f"{prefix}.{name}"]).reshape(b.shape))
        state32 = {f"{prefix}.{n}": v.float().clone() for prefix, m in modules.items() for n, v in m.named_parameters()}
        state32.update({f"{prefix}.{n}": v.float().clone() for prefix, m in modules.items() for n, v in m.named_buffers() if "running" in n})
    for m in modules.values():
        m.double().eval()

    g = torch.Generator().manual_seed(227)
    if stored is None:
        tokens = torch.tensor([[3, 7, 1, 12, 19, 5, 0, 0]])
        x16 = torch.randn(1, 10, 16, generator=g)
        x16[:, 8:] = 0
        seq = torch.randn(1, 7, 16, generator=g)
        mel = torch.randn(1, 20, 20, generator=g)
    else:
        tokens = torch.tensor([stored["tokens"]])
        x16 = torch.tensor(stored["x16"]).reshape(1, 10, 16)
        seq = torch.tensor(stored["seq"]).reshape(1, 7, 16)
        mel = torch.tensor(stored["mel"]).reshape(1, 20, 20)
    with torch.no_grad():
        enc = encoder(tokens)
        dur = duration(enc, tokens == 0)
        conv = blocks(x16)
        # The prosody language model's stack: positions as FFTBlocks adds them, then causal decoder layers.
        pos = SinusoidalPositionalEmbedding(16, 0, init_size=64).double()(seq[..., 0])
        h = (seq + pos).transpose(0, 1)
        mask = torch.triu(torch.full((7, 7), float("-inf")), 1)
        for layer in decoder:
            h, _ = layer(h, self_attn_mask=mask)
        dec = h.transpose(0, 1)
        validity, starts, _ = disc(mel[:, None], torch.tensor([20]), start_frames_wins=[[3], [1]])
    torch.set_default_dtype(torch.float32)
    data = {"tokens": tokens[0].tolist(), "x16": x16.reshape(-1).tolist(), "seq": seq.reshape(-1).tolist(), "mel": mel.reshape(-1).tolist(),
            "encoder": enc.reshape(-1).tolist(), "duration": dur.reshape(-1).tolist(), "blocks": conv.reshape(-1).tolist(),
            "decoder": dec.reshape(-1).tolist(), "positions": pos.reshape(-1).tolist(), "validity": validity.reshape(-1).tolist(), "starts": [3, 1]}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, len(state32), "tensors; validity", data["validity"])
        return
    check = _fixture.Comparison("Mega-TTS modules")
    for key in ("encoder", "duration", "blocks", "decoder", "validity"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
