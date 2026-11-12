"""WhisperSpeech S2A reference (collabora/WhisperSpeech s2a_delar_mup_wds_mlang.py SADelARTransformer, and the
conditioned variant s2a_delar_mup_wds_mlang_cond.py): tiny models (depth 2 — one encoder and three decoder layers — 2
heads of 8, ffn ×4, 3 quantizers over EnCodec's 1024 codes plus 2 specials, 10 semantic positions over 8 codes + 1
padding at width 4, a 6-wide speaker embedding, 30 acoustic positions), with the default checkpoint's tunables (rope,
the hidden-to-embedding projection kept) and seeded random weights in float64 (LayerNorm's and the unembedding's
float32 casts and the rotary table's float32 frequencies removed).

Records for each: the training loss (each quantizer's next-token cross-entropy, ignoring 1024, averaged), the eval
logits, and greedy generation (T = 1e-5, N = 18) through generate() with optimize()'s caches and merged projections.
Usage: python tools/reference-data/whisperspeech_s2a_reference.py --whisperspeech <WhisperSpeech checkout> [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402
from whisperspeech_t2s_reference import load as load_t2s  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/whisperspeech_s2a_reference.json"
CONFIG = dict(depth=2, ctx_n=30, stoks_len=10, stoks_codes=9, stoks_width=4, spk_width=6, n_head=2, head_width=8, ffn_mult=4, quantizers=3)


def patch_rotary(modules):
    def forward(self, x, seq_dim=1):
        if not self.seq_len_cached:
            dim = self.inv_freq.shape[0] * 2
            inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            t = torch.arange(2500, dtype=torch.float64)
            emb = torch.cat((torch.outer(t, inv), torch.outer(t, inv)), dim=-1)
            self.seq_len_cached = 2500
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached
    modules.Rotary.forward = forward


def build(cls, tunables_cls):
    torch.manual_seed(0)
    model = cls(**CONFIG, tunables=tunables_cls(**tunables_cls.upgrade({"rope": True, "linear_heads": False}))).double()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.2 * torch.randn_like(p) if "ln" in name and name.endswith("weight") else 0.4 * torch.randn_like(p))
    return model


def run(model, conds_fn):
    stoks = torch.tensor([3, 1, 4, 1, 5, 7, 2, 6])
    pad = CONFIG["stoks_codes"] - 1
    in_stoks = torch.nn.functional.pad(stoks, (1, CONFIG["stoks_len"] - len(stoks) - 1), value=pad)[None]
    q, n = CONFIG["quantizers"], 14
    torch.manual_seed(5)
    atoks = torch.randint(0, 1024, (1, q, n))
    for i in range(q):
        atoks[0, i, :i + 1] = 1025          # the delay's start tokens
    atoks[0, 1, -1] = 1024                    # an ignored target
    speaker = torch.randn(1, CONFIG["spk_width"])
    model.train()
    _, loss = model(in_stoks, atoks, conds_fn(speaker))
    model.eval()
    with torch.no_grad():
        logits = model(in_stoks, atoks, conds_fn(speaker), noloss=True)
        model.optimize(max_batch_size=1, dtype=torch.float64, torch_compile=False)
        generated = model.generate(stoks, speaker[0], N=18, T=1e-5, show_progress_bar=False)
    return {"stoks": stoks.tolist(), "atoks": atoks.reshape(-1).tolist(), "speaker": speaker.reshape(-1).tolist(), "loss": float(loss),
            "logits": logits.reshape(-1).tolist(), "generated": generated[0].reshape(-1).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisperspeech")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    load_t2s(args.whisperspeech)
    import whisperspeech.modules as modules
    patch_rotary(modules)
    from whisperspeech import s2a_delar_mup_wds_mlang as s2a, s2a_delar_mup_wds_mlang_cond as s2a_cond

    plain = build(s2a.SADelARTransformer, s2a.Tunables)
    plain_state = {k: v.reshape(-1).tolist() for k, v in plain.state_dict().items() if torch.is_tensor(v) and v.dtype.is_floating_point}
    plain_data = run(plain, lambda speaker: speaker)
    cond = build(s2a_cond.SADelARTransformer, s2a_cond.Tunables)
    cond_state = {k: v.reshape(-1).tolist() for k, v in cond.state_dict().items() if torch.is_tensor(v) and v.dtype.is_floating_point}
    cond_data = run(cond, lambda speaker: [dict(speaker=speaker[0], snr=25.0, c50=33.0)])
    data = {"plain": plain_data, "cond": cond_data}
    if args.write:
        _fixture.save(FIXTURE, {"config": CONFIG, "plain_state": plain_state, "cond_state": cond_state, **data}, __file__)
        print("wrote", FIXTURE, plain_data["loss"], cond_data["loss"], plain_data["generated"][:12])
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("WhisperSpeech S2A")
    for key in ("plain", "cond"):
        check.floats(f"{key} loss", [stored[key]["loss"]], [data[key]["loss"]], 1e-12, float)
        check.floats(f"{key} logits", stored[key]["logits"], data[key]["logits"], 1e-12, float)
        if stored[key]["generated"] != data[key]["generated"]:
            check.failures.append(f"{key} generation differs")
    check.report(exact=False)


if __name__ == "__main__":
    main()
