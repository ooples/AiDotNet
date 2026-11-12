"""Reference outputs of Amphion's released FACodec (amphion/naturalspeech3_facodec) in float64: v1's encoder, factorized
quantizers, timbre extractor, supervision and gradient-reversal heads and decoder; v2's mel prosody branch; and the
redecoder's voice conversion.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -v <facodec dir>:/ns3 -w /ref/Amphion-full gptsovits-ref \
        python /repo/tools/reference-data/facodec_checkpoint_reference.py

<facodec dir> holds ns3_facodec_{encoder,decoder,encoder_v2,decoder_v2,redecoder}.bin; writes facodec_checkpoint_reference.json
there.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
# melspec.py imports pyworld, parselmouth and torchaudio's pitch_shift for helpers FACodec never calls.
from types import ModuleType  # noqa: E402
for _name in ("pyworld", "parselmouth", "torchaudio", "torchaudio.functional"):
    sys.modules.setdefault(_name, ModuleType(_name))
sys.modules["torchaudio.functional"].pitch_shift = None
sys.modules["torchaudio"].functional = sys.modules["torchaudio.functional"]

from models.codec.ns3_codec import FACodecDecoder, FACodecDecoderV2, FACodecEncoder, FACodecEncoderV2, FACodecRedecoder  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


DECODER = dict(in_channels=256, upsample_initial_channel=1024, ngf=32, up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
               vq_dim=256, codebook_dim=8, codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
               use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True)


def load(module, name):
    module.load_state_dict(torch.load(f"/ns3/{name}", map_location="cpu"))
    return module.double().eval()


def wave(f1, f2, n=16000):
    return torch.tensor([0.3 * math.sin(2 * math.pi * f1 * i / 16000) + 0.1 * math.sin(2 * math.pi * f2 * i / 16000) for i in range(n)]).view(1, 1, -1)


def main():
    torch.set_default_dtype(torch.float64)
    a, b = wave(180, 470), wave(240, 610, 12800)
    out = {"wave_a": tolist(a), "wave_b": tolist(b)}
    with torch.no_grad():
        encoder = load(FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256), "ns3_facodec_encoder.bin")
        decoder = load(FACodecDecoder(**DECODER), "ns3_facodec_decoder.bin")
        enc = encoder(a)
        emb, codes, _, quantized, spk = decoder(enc, eval_vq=False, vq=True)
        heads = decoder(None, vq=False, speaker_embedding=spk, quantized=quantized)
        out["v1"] = {"enc": tolist(enc), "emb": tolist(emb), "codes": codes[:, 0].tolist(), "quantized": [tolist(q) for q in quantized],
                     "spk": tolist(spk), "audio": tolist(decoder.inference(emb, spk)),
                     "f0": tolist(heads["f0"]), "uv": tolist(heads["uv"]), "phone_argmax": heads["phone"].argmax(-1)[0].tolist(),
                     "phone_head": tolist(heads["phone"][0, :, :8]), "res_f0": tolist(heads["res_f0"]), "res_uv": tolist(heads["res_uv"]),
                     "res_phone_head": tolist(heads["res_phone"][0, :, :8]), "x_timbre_head": tolist(heads["x_timbre"][0, :8]),
                     "x_timbre_argmax": int(heads["x_timbre"].argmax(-1)[0])}
        enc_b = encoder(b)
        _, codes_b, _, _, spk_b = decoder(enc_b, eval_vq=False, vq=True)

        redecoder = load(FACodecRedecoder(), "ns3_facodec_redecoder.bin")
        converted = redecoder.vq2emb(codes, spk_b, use_residual=False)
        out["redecoder"] = {"spk_b": tolist(spk_b), "emb": tolist(converted), "audio": tolist(redecoder.inference(converted, spk_b))}

        encoder2 = load(FACodecEncoderV2(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256), "ns3_facodec_encoder_v2.bin")
        decoder2 = load(FACodecDecoderV2(**DECODER), "ns3_facodec_decoder_v2.bin")
        enc2 = encoder2(a)
        prosody = encoder2.get_prosody_feature(a)
        emb2, codes2, _, _, spk2 = decoder2(enc2, prosody, eval_vq=False, vq=True)
        out["v2"] = {"enc": tolist(enc2), "prosody": tolist(prosody), "codes": codes2[:, 0].tolist(), "spk": tolist(spk2),
                     "audio": tolist(decoder2.inference(emb2, spk2))}
    path = "/ns3/facodec_checkpoint_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes; frames", codes.shape[-1])


if __name__ == "__main__":
    main()
