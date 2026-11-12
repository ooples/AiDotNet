"""Reference fixture for NaturalSpeech 3's FACodec at a tiny configuration in float64, from Amphion's own modules
(models/codec/ns3_codec) with random weights: the encoder, the factorized quantizers, the timbre extractor, the supervision
and gradient-reversal heads, the decoder and the redecoder. The decoder hard-codes a 256-wide timbre Transformer and
5003/245200-way heads; those are replaced by tiny ones of the same classes.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/Amphion-full gptsovits-ref \
        python /repo/tools/reference-data/facodec_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/facodec_reference.json.
"""
import json
import math
import os
import sys
from types import ModuleType

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
for _name in ("pyworld", "parselmouth", "torchaudio", "torchaudio.functional"):
    sys.modules.setdefault(_name, ModuleType(_name))
sys.modules["torchaudio.functional"].pitch_shift = None
sys.modules["torchaudio"].functional = sys.modules["torchaudio.functional"]

from models.codec.ns3_codec import FACodecDecoder, FACodecEncoder, FACodecRedecoder  # noqa: E402
from models.codec.ns3_codec.facodec import CNNLSTM  # noqa: E402
from models.codec.ns3_codec.gradient_reversal import GradientReversal  # noqa: E402
from models.codec.ns3_codec.transformer import TransformerEncoder  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


D, PHONES, SPEAKERS = 16, 12, 7


def tiny_transformer(cln):
    return TransformerEncoder(enc_emb_tokens=None, encoder_layer=1, encoder_hidden=D, encoder_head=2, conv_filter_size=32,
                              conv_kernel_size=5, encoder_dropout=0.1, use_cln=cln)


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("weight_g"):
                p.copy_(p.abs() + 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.3)


def main():
    torch.manual_seed(11)
    torch.set_default_dtype(torch.float64)
    encoder = FACodecEncoder(ngf=4, up_ratios=[2, 2], out_channels=D).double()
    decoder = FACodecDecoder(in_channels=D, upsample_initial_channel=16, ngf=4, up_ratios=[2, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
                             vq_dim=D, codebook_dim=4, codebook_size_prosody=4, codebook_size_content=4, codebook_size_residual=4,
                             use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True).double()
    decoder.timbre_encoder = tiny_transformer(False)
    decoder.phone_predictor = CNNLSTM(D, PHONES, 1)
    decoder.res_phone_predictor = nn.Sequential(GradientReversal(alpha=1.0), CNNLSTM(D, PHONES, 1))
    decoder.x_timbre_predictor = nn.Sequential(GradientReversal(alpha=1), CNNLSTM(D, SPEAKERS, 1, global_pred=True))
    redecoder = FACodecRedecoder(in_channels=D, upsample_initial_channel=16, up_ratios=[2, 2], vq_dim=D, codebook_size_prosody=4,
                                 codebook_size_content=4, codebook_size_residual=4).double()
    redecoder.timbre_cond_prosody_enc = tiny_transformer(True)
    for m in (encoder, decoder, redecoder):
        m.double()
        randomize(m)
        m.eval()

    n = 64
    a = torch.tensor([0.3 * math.sin(2 * math.pi * 0.07 * i) + 0.2 * math.sin(2 * math.pi * 0.19 * i) for i in range(n)]).view(1, 1, -1)
    b = torch.tensor([0.4 * math.sin(2 * math.pi * 0.11 * i) for i in range(48)]).view(1, 1, -1)
    out = {"wave_a": tolist(a), "wave_b": tolist(b)}
    with torch.no_grad():
        enc = encoder(a)
        emb, codes, _, quantized, spk = decoder(enc, eval_vq=False, vq=True)
        torch.manual_seed(0)
        import numpy as np
        np.random.seed(0)
        decoder.use_random_mask_residual = False
        heads = decoder(None, vq=False, speaker_embedding=spk, quantized=quantized)
        out["enc"] = tolist(enc)
        out["codes"] = codes[:, 0].tolist()
        out["quantized"] = [tolist(q) for q in quantized]
        out["spk"] = tolist(spk)
        out["emb"] = tolist(emb)
        out["audio"] = tolist(decoder.inference(emb, spk))
        out["vq2emb_no_detail"] = tolist(decoder.vq2emb(codes, use_residual_code=False))
        for key in ("f0", "uv", "phone", "res_f0", "res_uv", "res_phone", "x_timbre"):
            out[key] = tolist(heads[key])
        _, codes_b, _, _, spk_b = decoder(encoder(b), eval_vq=False, vq=True)
        converted = redecoder.vq2emb(codes, spk_b, use_residual=False)
        out["spk_b"] = tolist(spk_b)
        out["redecoder_emb"] = tolist(converted)
        out["redecoder_audio"] = tolist(redecoder.inference(converted, spk_b))
    state = {}
    for prefix, module in (("encoder.", encoder), ("decoder.", decoder), ("redecoder.", redecoder)):
        for k, v in module.state_dict().items():
            if not k.endswith("position_emb.pe"):
                state[prefix + k] = tolist(v)
    out["state"] = state
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "facodec_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes; frames", codes.shape[-1])


if __name__ == "__main__":
    main()
