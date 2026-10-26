"""MegaTTS 3's inference glue through the release's own functions (bytedance/MegaTTS3, tts/frontend_function.py:
make_dur_prompt, dur_pred and prepare_inputs_for_dit), with the duration model of megatts3_duration_reference.py (its
tiny configuration and stored weights).

A prompt of seven phones (Chinese tones and English phones, a pause among them) with random durations; make_dur_prompt's
duration tokens; dur_pred for two chunks (the first, then the final; pauses among their phones), torch.rand_like replaced by
recorded float32 draws in its call order; prepare_inputs_for_dit with a random prompt latent of 4 channels. Precision
patches as in megatts3_duration_reference.py (float64, the rotary tables and the attention's softmax). Records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_pipeline_reference.json the inputs, the draws, each chunk's
frame-to-phone map and the transformer's inputs.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-outetts environment (openai-whisper,
librosa).
"""
import argparse
import os
import sys
import types

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_pipeline_reference.json"
DURATION = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_duration_reference.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    import tts.modules.ar_dur.commons.rot_transformer as rot
    from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
    from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
    import tts.frontend_function as frontend

    def create(self, *, width, length):
        theta = torch.pow(10000, -torch.arange(0, width, 2, dtype=torch.float64) / width)
        m_theta = torch.arange(length, dtype=torch.float64).unsqueeze(1) * theta.unsqueeze(0)
        m_theta = torch.cat([m_theta, m_theta], dim=-1)
        self.register_buffer("cos", m_theta.cos().view([length, width]), persistent=False)
        self.register_buffer("sin", m_theta.sin().view([length, width]), persistent=False)
    rot.RotaryEmbeddings._create_rotary_embed = create
    rot.softmax = lambda x, dim: torch.softmax(x, dim=dim)

    duration = _fixture.load(DURATION)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    hp = dict(duration["hp"])
    model = ARDurPredictor(hp, hp["hidden_size"], duration["dec"], duration["layers"], duration["phones"], duration["codes"], use_rot_embed=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.tensor(duration["state"][name], dtype=torch.float32).reshape(p.shape))
    model = model.double().eval()
    fake = types.SimpleNamespace(dur_model=model, hp_dur_model={"dur_code_size": duration["codes"]}, fm=8, length_regulator=LengthRegulator(),
                                 device="cpu", precision=torch.float32, cfg_mask_token_phone=301, cfg_mask_token_tone=31, vae_stride=4)

    g = torch.Generator().manual_seed(193)
    if args.write:
        ph_ref = [57, 12, 148, 120, 33, 163, 9]
        tone_ref = [4, 3, 3, 11, 13, 3, 15]
        dur_ref = torch.randint(2, 14, (7,), generator=g).tolist()
        chunks = [([145, 40, 77, 163, 101], [3, 4, 12, 3, 14]), ([88, 23, 153], [13, 3, 3])]
    else:
        ph_ref, tone_ref, dur_ref = stored["ph_ref"], stored["tone_ref"], stored["dur_ref"]
        chunks = [(c["ph"], c["tone"]) for c in stored["chunks"]]
    mel2ph_ref = fake.length_regulator(torch.LongTensor([dur_ref]))
    mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1) // 8 * 8]
    ph_ref_t, tone_ref_t = torch.LongTensor([ph_ref]), torch.LongTensor([tone_ref])

    draws_out = []
    real_rand_like = torch.rand_like

    def rand_like(x):
        if stored is None:
            v = torch.rand(x.shape, generator=g, dtype=torch.float32)
        else:
            v = torch.tensor(stored["draws"][len(draws_out)], dtype=torch.float32).reshape(x.shape)
        draws_out.append(v.reshape(-1).tolist())
        return v
    torch.rand_like = rand_like
    with torch.no_grad():
        state, ctx = frontend.make_dur_prompt(fake, mel2ph_ref, ph_ref_t, tone_ref_t)
        frames = mel2ph_ref.size(1) // 4
        latent = (torch.randn(1, frames, 4, generator=g) if stored is None else torch.tensor(stored["latent"]).reshape(1, frames, 4)).float().double()
        results = []
        for i, (ph, tone) in enumerate(chunks):
            ph_t, tone_t = torch.LongTensor([ph]), torch.LongTensor([tone])
            mel2ph = frontend.dur_pred(fake, ctx, state, ph_t, tone_t, i, 0.1, 1.0, is_first=i == 0, is_final=i == len(chunks) - 1)
            inputs = frontend.prepare_inputs_for_dit(fake, mel2ph_ref, mel2ph, ph_ref_t, tone_ref_t, ph_t, tone_t, latent)
            results.append({"ph": ph, "tone": tone, "mel2ph": mel2ph[0].tolist(), "phone": inputs["phone"].tolist(), "tone_rows": inputs["tone"].tolist(),
                            "dur": inputs["dur"].tolist(), "lat_ctx": inputs["lat_ctx"].reshape(-1).tolist(), "ctx_mask": inputs["ctx_mask"].reshape(-1).tolist()})
    torch.rand_like = real_rand_like
    torch.set_default_dtype(torch.float32)
    data = {"ph_ref": ph_ref, "tone_ref": tone_ref, "dur_ref": dur_ref, "mel2ph_ref": mel2ph_ref[0].tolist(), "ctx": ctx[0].tolist(),
            "latent": latent.reshape(-1).tolist(), "draws": draws_out, "chunks": results}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "ctx", data["ctx"], "chunk frames", [len(c["mel2ph"]) for c in results])
        return
    check = _fixture.Comparison("MegaTTS 3 pipeline")
    check.exact("ctx", stored["ctx"], data["ctx"])
    for a, b in zip(stored["chunks"], results):
        for key in ("mel2ph", "phone", "tone_rows", "dur", "ctx_mask"):
            check.exact(key, a[key], b[key])
        check.floats("lat_ctx", a["lat_ctx"], b["lat_ctx"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
