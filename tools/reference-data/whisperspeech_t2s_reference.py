"""WhisperSpeech T2S reference (collabora/WhisperSpeech whisperspeech/t2s_up_wds_mlang_enclm.py TSARTransformer and modules.py):
a tiny model (depth 2 — one encoder and three decoder layers by the 0.25 encoder ratio — 2 heads of 8, ffn ×2, 40 UTF-8
byte tokens, 30 semantic positions over 32 codes + 1 padding, frozen semantic width 8, so both embedding projectors run),
seeded random weights in float64 (the LayerNorm's and unembed's float32 casts removed).

Records the training loss in train mode (causal encoder with its 0.1 auxiliary loss) on a text and semantic tokens,
the eval-mode logits, and greedy generation (T = 1e-5) for the text with a 2-token prompt, through the model's own
generate() (KV caches and merged projections from optimize(), the cps embedding, the language embedding).
Usage: python tools/reference-data/whisperspeech_t2s_reference.py --whisperspeech <WhisperSpeech checkout> [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/whisperspeech_t2s_reference.json"
CONFIG = dict(depth=2, n_head=2, head_width=8, ffn_mult=2, ttoks_len=40, ttoks_codes=256, stoks_len=30, stoks_codes=33, stoks_width=8)


def load(root):
    sys.path.insert(0, root)
    import whisperspeech.modules as modules
    import torch.nn.functional as F

    # float64 throughout: LayerNorm and the unembedding cast to float32.
    modules.LayerNorm.forward = lambda self, x: torch.nn.LayerNorm.forward(self, x)

    def unembed(self, embs):
        if not self.training and self.merged_out is not None:
            return F.linear(embs, self.merged_out, self.bias_out)
        orig = embs
        if self.hidden_to_emb: embs = self.hidden_to_emb(embs)
        main = embs @ self.main.weight.to(embs.dtype).T
        if not self.special_codes: return main
        return torch.cat([main, orig @ self.special.weight.to(orig.dtype).T], dim=-1)

    modules.FlexEmbeddings.unembed = unembed
    from whisperspeech import t2s_up_wds_mlang_enclm as t2s
    return t2s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisperspeech")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    t2s = load(args.whisperspeech)
    torch.manual_seed(0)
    tunables = t2s.Tunables(**t2s.Tunables.upgrade({}))
    model = t2s.TSARTransformer(**CONFIG, tunables=tunables).double()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("ln.weight") or name.endswith("ln_post.weight") or "_ln." in name and name.endswith("weight"):
                p.copy_(1 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.4 * torch.randn_like(p))
    state = {k: v.reshape(-1).tolist() for k, v in model.state_dict().items()}

    text = "Hi there"
    tok = t2s.CharTokenizer()
    ttoks = torch.tensor(tok.encode(text))
    in_ttoks = torch.nn.functional.pad(ttoks, (1, CONFIG["ttoks_len"] - len(ttoks) - 1), value=tok.eot)[None]
    out_ttoks = torch.nn.functional.pad(ttoks, (0, CONFIG["ttoks_len"] - len(ttoks)), value=tok.eot)[None]
    stoks = torch.tensor([3, 17, 5, 30, 2, 2, 9, 11, 21, 0, 7, 14])
    pad = CONFIG["stoks_codes"] - 1
    in_stoks = torch.nn.functional.pad(stoks, (1, CONFIG["stoks_len"] - len(stoks) - 1), value=pad)[None]
    out_stoks = torch.nn.functional.pad(stoks, (0, CONFIG["stoks_len"] - len(stoks)), value=pad)[None]
    langs, cpss = torch.tensor([0]), torch.tensor([14.0])
    positions = torch.arange(CONFIG["stoks_len"])

    model.train()
    _, loss = model(in_ttoks, out_ttoks, langs, cpss, in_stoks, out_stoks, in_stoks_positions=positions)
    model.eval()
    with torch.no_grad():
        logits, _ = model(in_ttoks, out_ttoks, langs, cpss, in_stoks, out_stoks, in_stoks_positions=positions)
        model.optimize(max_batch_size=1, dtype=torch.float64, torch_compile=False)
        generated = model.generate(text, cps=14, lang="en", stoks_prompt=torch.tensor([3, 17]), N=20, T=1e-5, show_progress_bar=False)
    data = {"loss": float(loss), "logits": logits.reshape(-1).tolist(), "generated": generated[0].tolist()}
    if args.write:
        _fixture.save(FIXTURE, {"config": CONFIG, "text": text, "stoks": stoks.tolist(), "state": state, **data}, __file__)
        print("wrote", FIXTURE, float(loss), generated[0].tolist())
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("WhisperSpeech T2S")
    check.floats("loss", [stored["loss"]], [data["loss"]], 1e-12, float)
    check.floats("logits", stored["logits"], data["logits"], 1e-12, float)
    if stored["generated"] != data["generated"]:
        check.failures.append("generation differs")
    check.report(exact=False)


if __name__ == "__main__":
    main()
