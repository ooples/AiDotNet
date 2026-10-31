"""Moshi RQ-Transformer reference: the release's LMModel (kyutai-labs/moshi, moshi/models/lm.py) at a tiny size, in float64.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_lm_reference.json:
  - a tiny seeded LMModel (2 + 2 layers, the paper's 17-stream layout shrunk to 1 text + 2 speakers × 3 codebooks, the
    Depth Transformer predicting all 6 audio streams with weights per step), its state, and for a code sequence with absent
    (−1) values: the training text logits and audio logits (LMModel.forward, undelayed) with their masks;
  - the same with dep_q = 3 (the release's layout: only Moshi's streams predicted);
  - the Eq. 7 loss of the 17-stream model: text CE with PAD/EPAD weighted 0.5 plus the audio CE with each speaker's
    semantic stream weighted 100, each normalized by its weights (moshi-finetune's compute_loss_with_mask generalized to α_k).
The release computes RMSNorm (rms_norm_f32) and the rotary embedding in float32; both are patched to float64 here.
Default: verify; --write: regenerate.
"""
import base64
import math
import os
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/moshi_lm_reference.json"
CONFIG = dict(n_q=6, card=12, text_card=20, dim=16, num_heads=2, hidden_scale=4.125, norm="rms_norm_f32", depformer_dim=8,
              depformer_dim_feedforward=int(4.125 * 8), depformer_num_heads=2, depformer_num_layers=2, depformer_multi_linear=True,
              depformer_weights_per_step=True, depformer_pos_emb="none", num_layers=2, causal=True, context=5, max_period=10000,
              gating="silu", positional_embedding="rope", depformer_gating="silu", existing_text_padding_id=3)
DELAYS = [0, 0, 1, 1, 0, 1, 1]
CODES = [[5, 3, 3, 0, 7, 3, -1, 9],
         [1, 4, 4, 2, 11, 0, 6, 6],
         [2, 2, 8, 5, 1, 10, 3, 7],
         [0, 9, 3, -1, 4, 4, 2, 8],
         [7, 7, 1, 6, 6, 0, 11, 2],
         [3, 5, 9, 9, 2, 1, 4, -1],
         [10, 0, 2, 7, 8, 8, 5, 3]]
USER = [[4, 1, 7, 7, 2, 9, 0, 3, 11, 5],
        [8, 8, 3, 0, 6, 1, 10, 2, 2, 7],
        [1, 5, 5, 9, 4, 3, 8, 0, 6, 6]]


def patch_float64():
    from moshi.modules import rope, transformer

    def apply_rope(q, k, offset, max_period=10_000, interleave=True, time_before_heads=False):
        B, H, T, D = q.shape
        ds = torch.arange(D // 2, dtype=torch.float64)
        freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
        ts = offset.double().view(-1, 1) + torch.arange(T, dtype=torch.float64)
        ts = ts.view(B, 1, -1, 1)
        rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
        q = q.view(*q.shape[:-1], D // 2, 2)
        k = k.view(*k.shape[:-1], D // 2, 2)
        qr, qi, kr, ki = q[..., 0], q[..., 1], k[..., 0], k[..., 1]
        qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1).view(*q.shape[:-2], D)
        ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1).view(*k.shape[:-2], D)
        return qo, ko

    rope.apply_rope = apply_rope
    transformer.RMSNorm.forward = lambda self, x: _rms(x, self.alpha, self.eps)


def _rms(x, alpha, eps):
    var = eps + torch.mean(x ** 2, dim=2, keepdim=True)
    return x * (alpha.to(var) * torch.rsqrt(var))


def build(dep_q, seed, stored, key):
    from moshi.models.lm import LMModel
    torch.manual_seed(seed)
    model = LMModel(delays=DELAYS, dep_q=dep_q, **CONFIG)
    if stored:
        model.load_state_dict(load_safetensors(base64.b64decode(stored[key])))
    return model.double().eval()


def main():
    args = _fixture.parse_args(__doc__)
    torch.set_default_dtype(torch.float64)
    patch_float64()
    stored = None if args.write else _fixture.load(FIXTURE)
    data = {}
    states = {}
    codes = torch.tensor([CODES])
    for name, dep_q, seed in (("paper", 6, 2294), ("release", 3, 2295)):
        model = build(dep_q, seed, stored, f"{name}_state")
        with torch.no_grad():
            out = model(codes)
        data[f"{name}_text_logits"] = [float(v) for v in torch.nan_to_num(out.text_logits[0, 0], nan=0.0).reshape(-1).tolist()]
        data[f"{name}_text_mask"] = [bool(v) for v in out.text_mask[0, 0].tolist()]
        data[f"{name}_audio_logits"] = [float(v) for v in torch.nan_to_num(out.logits[0], nan=0.0).reshape(-1).tolist()]
        data[f"{name}_audio_mask"] = [[bool(v) for v in row] for row in out.mask[0].tolist()]
        states[f"{name}_state"] = base64.b64encode(save_safetensors({k: v.contiguous() for k, v in model.state_dict().items()})).decode("ascii")
        # Greedy streaming (LMGen, use_sampling=False) for 10 steps, past the context of 5: the release's layout reads the
        # user's 3 streams from USER (one frame a step); the paper's predicts every stream.
        from moshi.models.lm import LMGen
        gen = LMGen(model, use_sampling=False)
        frames, text_logits = [], []
        gen.on_text_logits_hook = lambda logits: text_logits.append([float(v) for v in logits.reshape(-1).tolist()])
        with torch.no_grad(), gen.streaming(1):
            for step in range(10):
                user = torch.tensor([[[USER[q][step]] for q in range(3)]]) if dep_q == 3 else torch.zeros(1, 0, 1, dtype=torch.long)
                step_out = gen.step(user)
                frames.append(None if step_out is None else [int(v) for v in step_out[0, :, 0].tolist()])
        data[f"{name}_stream_frames"] = frames
        data[f"{name}_stream_text_logits"] = [v for row in text_logits for v in row]
        if name == "paper":
            # Eq. 7: text CE (PAD 3 and EPAD 0 weighted 0.5) + audio CE with α = 100 on streams 0 and 3 (each speaker's semantic).
            tl = out.text_logits[0, 0]
            tm = out.text_mask[0, 0]
            target = codes[0, 0].clamp(min=0)
            w = tm.double() * torch.where((codes[0, 0] == 3) | (codes[0, 0] == 0), 0.5, 1.0)
            ce = torch.nn.functional.cross_entropy(torch.nan_to_num(tl), target, reduction="none")
            text_loss = (ce * w).sum() / w.sum()
            al = torch.nan_to_num(out.logits[0])
            am = out.mask[0]
            at = codes[0, 1:].clamp(min=0)
            alpha = torch.tensor([100.0 if s % 3 == 0 else 1.0 for s in range(6)]).view(6, 1)
            aw = am.double() * alpha
            ace = torch.nn.functional.cross_entropy(al.reshape(-1, 12), at.reshape(-1), reduction="none").view(6, -1)
            data["paper_loss"] = float(text_loss + (ace * aw).sum() / aw.sum())
    if args.write:
        _fixture.save(FIXTURE, {"config": CONFIG, "delays": DELAYS, "codes": CODES, "user": USER, **states, **data}, __file__)
        print("wrote", FIXTURE, data["paper_loss"])
        return
    check = _fixture.Comparison("Moshi LM")
    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], float):
            check.floats(k, stored[k], v, 1e-12, float)
        elif isinstance(v, list) and v and v[0] is None or isinstance(v, list) and v and isinstance(v[0], list) and v[0] and isinstance(v[0][0], int):
            if stored[k] != v:
                check.failures.append(f"{k}: differs")
        elif isinstance(v, float):
            check.floats(k, [stored[k]], [v], 1e-12, float)
        elif stored[k] != v:
            check.failures.append(f"{k}: differs")
    check.report(exact=False)


if __name__ == "__main__":
    main()
