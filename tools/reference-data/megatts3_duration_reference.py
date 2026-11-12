"""MegaTTS 3's autoregressive duration predictor through the release's own code (bytedance/MegaTTS3,
tts/modules/ar_dur/ar_dur_predictor.py with its relative-attention encoder and rotary decoder layers), on a tiny model.

ARDurPredictor takes its sizes from hparams; the test sets the release's duration_lm/config.yaml switches (lm_use_enc,
the tone label, phone positions, ar_mse, pre-norm, no speaker inputs) at width 16 (decoder 32, eight heads of 4) with two
encoder and two decoder layers and a small character vocabulary (the character encoder is unused at inference). Random
weights, float64. Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_duration_reference.json:
- the prompt pass of frontend_function.make_dur_prompt (teacher-forced on the prompt's duration tokens, building the
  incremental state) and dur_pred's continuation from it (first_decoder_inp, absolute query positions), the raw durations;
- a full (training) forward with the causal mask and the positions, its outputs.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-kani environment.
"""
import argparse
import os
import sys
from copy import deepcopy

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_duration_reference.json"
HP = dict(hidden_size=16, ffn_hidden_size=32, num_heads=2, enc_layers=2, enc_ffn_kernel_size=3, dropout=0.0, enc_prenet=True,
          enc_pre_ln=True, encoder_type="rel_fft", lm_use_enc=True, mega_use_char=True, char_dict_size=50, use_ph_pos_embed=True,
          ling_labels=["tone"], ling_label_dict_size=[20], use_pos_embed=False, use_post_ln=False, use_spk_embed=False,
          use_spk_id=False, use_spk_enc=False, dur_model_type="ar_mse", max_tokens=6000, lm_bias=True)
DEC, LAYERS, PHONES, CODES = 32, 2, 174, 128


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    import tts.modules.ar_dur.commons.rot_transformer as rot
    from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor

    # Precision patch: the rotary tables are built in float32; the same in float64.
    def create(self, *, width, length):
        theta = torch.pow(10000, -torch.arange(0, width, 2, dtype=torch.float64) / width)
        m_theta = torch.arange(length, dtype=torch.float64).unsqueeze(1) * theta.unsqueeze(0)
        m_theta = torch.cat([m_theta, m_theta], dim=-1)
        self.register_buffer("cos", m_theta.cos().view([length, width]), persistent=False)
        self.register_buffer("sin", m_theta.sin().view([length, width]), persistent=False)
    rot.RotaryEmbeddings._create_rotary_embed = create
    # Precision patch: seq_utils.softmax computes in float32 (the attention's softmax); the same in float64.
    rot.softmax = lambda x, dim: torch.softmax(x, dim=dim)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(139)
    model = ARDurPredictor(dict(HP), HP["hidden_size"], DEC, LAYERS, PHONES, CODES, use_rot_embed=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name == "project_out_dim.0.bias":
                # Durations of a few frames, so the rounded outputs vary.
                p.copy_(torch.full_like(p, 4.0))
            elif name.endswith("gamma") or (name.endswith("weight") and p.dim() == 1):
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        state32 = {k: v.float().clone() for k, v in model.named_parameters()}
    model = model.double().eval()

    g = torch.Generator().manual_seed(149)
    if args.write:
        ph_ref = torch.randint(3, PHONES, (1, 7), generator=g)
        tone_ref = torch.randint(1, 16, (1, 7), generator=g)
        ctx_dur = torch.randint(1, CODES, (1, 7), generator=g)
        ph_pred = torch.randint(3, PHONES, (1, 5), generator=g)
        tone_pred = torch.randint(1, 16, (1, 5), generator=g)
    else:
        ph_ref, tone_ref, ctx_dur = (torch.tensor([stored[k]]) for k in ("ph_ref", "tone_ref", "ctx_dur"))
        ph_pred, tone_pred = torch.tensor([stored["ph_pred"]]), torch.tensor([stored["tone_pred"]])
    with torch.no_grad():
        # frontend_function.make_dur_prompt
        P = ctx_dur.shape[1]
        _, state = model.infer(ph_ref, {"tone": tone_ref}, None, None, None, ctx_vqcodes=ctx_dur,
                               spk_pos_ids_flat=torch.LongTensor([list(range(0, P))]), return_state=True)
        # frontend_function.dur_pred's call (before its post-processing)
        T = ph_pred.shape[1]
        raw, steps = model.infer(ph_pred, {"tone": tone_pred}, None, None, None, incremental_state=deepcopy(state),
                                 first_decoder_inp=ctx_dur[:, -1:], spk_pos_ids_flat=torch.LongTensor([list(range(P, P + T))]), return_probs=True)
        # A training forward: previous durations (BOS first), causal mask, positions 0..T-1.
        prev = torch.cat([torch.full((1, 1), CODES + 1), ctx_dur[:, :-1]], dim=1)
        full = model(ph_ref, {"tone": tone_ref}, None, None, None, prev, spk_pos_ids_flat=torch.arange(P)[None])
    torch.set_default_dtype(torch.float32)
    data = {"hp": HP, "dec": DEC, "layers": LAYERS, "phones": PHONES, "codes": CODES, "ph_ref": ph_ref[0].tolist(), "tone_ref": tone_ref[0].tolist(),
            "ctx_dur": ctx_dur[0].tolist(), "ph_pred": ph_pred[0].tolist(), "tone_pred": tone_pred[0].tolist(), "raw": raw[0].tolist(), "steps": steps.reshape(-1).tolist(),
            "full": full.reshape(-1).tolist()}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "raw", data["raw"], "full", tuple(full.shape))
        return
    check = _fixture.Comparison("MegaTTS 3 duration")
    check.exact("raw", stored["raw"], data["raw"])
    check.floats("steps", stored["steps"], data["steps"], 1e-12, float)
    check.floats("full", stored["full"], data["full"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
