"""Reference outputs of Amphion's released NaturalSpeech 2 (amphion/naturalspeech2_libritts) with EnCodec's 24 kHz
checkpoint, in float64: the prompt and query tokens of a reference's codes, inference's prior (rounded durations, the
condition), one denoiser call, a three-step Euler reverse diffusion and the decoded waveform.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -v <ns2 dir>:/ns2 -w /ref/Amphion-full gptsovits-ref \
        sh -c "pip install -q --no-deps encodec==0.1.1 unidecode && python /repo/tools/reference-data/naturalspeech2_checkpoint_reference.py"

<ns2 dir> holds pytorch_model.bin and encodec_24khz-d7cc33bc.th; writes naturalspeech2_checkpoint_reference.json there.
"""
import json
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.getcwd())
sys.modules.setdefault("torchaudio", SimpleNamespace())

import encodec  # noqa: E402

_orig_24khz = encodec.EncodecModel.encodec_model_24khz


def _local_24khz():
    model = _orig_24khz(pretrained=False)
    model.load_state_dict(torch.load("/ns2/encodec_24khz-d7cc33bc.th", map_location="cpu"))
    return model


encodec.EncodecModel.encodec_model_24khz = staticmethod(_local_24khz)

from models.tts.naturalspeech2.ns2 import NaturalSpeech2  # noqa: E402


def ns(d):
    return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def main():
    torch.set_default_dtype(torch.float64)
    cfg = json.load(open("config/ns2.json"))["model"]
    cfg["diffusion"]["diffusion_type"] = "diffusion"   # egs/tts/NaturalSpeech2/exp_config_base.json
    model = NaturalSpeech2(ns(cfg))
    model.load_state_dict(torch.load("/ns2/pytorch_model.bin", map_location="cpu"))
    model = model.double().eval()
    codec = _local_24khz().double().eval()
    codec.set_target_bandwidth(12.0)

    # A 1.2 s two-tone reference, encoded at 12 kbps.
    n = 28800
    wave = torch.tensor([0.3 * math.sin(2 * math.pi * 180 * i / 24000) + 0.1 * math.sin(2 * math.pi * 470 * i / 24000) for i in range(n)])
    out = {"reference_audio": wave.tolist()}
    with torch.no_grad():
        frames = codec.encode(wave.view(1, 1, -1))
        ref_code = torch.cat([f[0] for f in frames], dim=-1)                                   # [1, 16, T']
        ref_mask = torch.ones(1, ref_code.shape[-1])
        out["ref_code"] = ref_code[0].tolist()
        from text.cmudict import valid_symbols
        symbols = valid_symbols + ["sp", "spn", "sil", "<s>", "</s>"]
        phone_id = torch.tensor([[symbols.index(p) for p in "HH AH0 L OW1 sp W ER1 L D".split()]])
        out["phone_id"] = phone_id[0].tolist()

        ref_latent = model.code_to_latent(ref_code)
        ref_hidden = model.prompt_encoder(model.prompt_lin(ref_latent.transpose(1, 2)), ref_mask, condition=None)
        spk_emb = ref_hidden.transpose(1, 2)
        query = model.query_emb(torch.arange(model.query_emb_num)).repeat(1, 1, 1)
        query, _ = model.query_attn(query, spk_emb.transpose(1, 2), spk_emb.transpose(1, 2), key_padding_mask=~(ref_mask.bool()))
        prior = model.prior_encoder(phone_id=phone_id, duration=None, pitch=None, phone_mask=None, mask=None, ref_emb=spk_emb,
                                    ref_mask=ref_mask, is_inference=True)
        cond = prior["prior_out"]
        out["prompt"] = tolist(ref_hidden)
        out["query"] = tolist(query)
        out["dur_pred_log"] = tolist(prior["dur_pred_log"])
        out["dur_pred_round"] = prior["dur_pred_round"][0].tolist()
        out["condition"] = tolist(cond)

        torch.manual_seed(5)
        t_frames = cond.shape[1]
        xt = torch.randn(1, 128, t_frames)
        out["xt"] = tolist(xt)
        out["x0_pred"] = tolist(model.diffusion.diff_estimator(xt, None, cond, torch.tensor([0.5]), query))
        start = torch.randn(1, 128, t_frames) / 1.2
        x0 = model.diffusion.reverse_diffusion(start, None, cond, 3, query)
        out["start"] = tolist(start)
        out["x0"] = tolist(x0)
        out["audio"] = tolist(codec.decoder(x0))

    path = "/ns2/naturalspeech2_checkpoint_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes;", "frames", t_frames, "durations", out["dur_pred_round"])


if __name__ == "__main__":
    main()
