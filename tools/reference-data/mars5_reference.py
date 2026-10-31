"""Reference fixture for MARS5-TTS (Camb-ai/MARS5-TTS at 84697c8): the minbpe tokenizers, the AR CodecLM and the NAR
ResidualTransformer at tiny sizes, the multinomial diffusion (schedule, posterior, loss, RePaint inference with its
random draws replaced by closed-form patterns), the AR logit processors and the silence trimmer. The RePaint forward
step's ramp is taken along the sequence (scaled_intent below), fixing the release's broadcast onto the codebook axis.

Runs the release's own modules in float64 in the gptsovits-ref image (torch, numpy, regex):

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/MARS5-TTS gptsovits-ref \
        python /repo/tools/reference-data/mars5_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/mars5_reference.json.
"""
import io
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())

from mars5 import diffuser  # noqa: E402
from mars5.diffuser import DSH, MultinomialDiffusion, get_schedule, perform_simple_inference  # noqa: E402
from mars5.minbpe.codebook import CodebookTokenizer  # noqa: E402
from mars5.minbpe.regex import GPT4_SPLIT_PATTERN, RegexTokenizer  # noqa: E402
from mars5.model import CodecLM, ResidualTransformer  # noqa: E402
from mars5.samplers import apply_typical_p, early_eos_penalty, freq_rep_penalty, top_k_top_p_filtering  # noqa: E402
from mars5.trim import trim  # noqa: E402


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def model_text(tok):
    buffer = io.StringIO()
    buffer.write("minbpe v1\n")
    buffer.write(f"{tok.pattern}\n")
    buffer.write(f"{len(tok.special_tokens)}\n")
    for special, idx in tok.special_tokens.items():
        buffer.write(f"{special} {idx}\n")
    for idx1, idx2 in tok.merges:
        buffer.write(f"{idx1} {idx2}\n")
    return buffer.getvalue()


def tokenizers(out):
    corpus = ("The quick brown fox jumps over the lazy dog. It's 12345 o'clock, isn't it? "
              "Hello world!  Hello   there,\nnew line\tand tabs. Ünïcödé text ½ and 漢字 too. ") * 3
    text = RegexTokenizer(GPT4_SPLIT_PATTERN)
    text.train(corpus, 300)
    text.register_special_tokens({"<|startoftext|>": 300, "<|endoftext|>": 301})
    text.special_tokens = {"<|startoftext|>": 300, "<|endoftext|>": 301}
    g = torch.Generator().manual_seed(4)
    codes = torch.randint(0, 16, (400,), generator=g)
    codes = torch.cat([codes[:200], codes[:100], codes[200:]])
    speech = CodebookTokenizer(GPT4_SPLIT_PATTERN)
    speech.train(" ".join(str(c) for c in codes.tolist()), 1024 + 40)
    speech.register_special_tokens({"<|endofspeech|>": 1064, "<|pad|>": 1065})
    speech.special_tokens = {"<|endofspeech|>": 1064, "<|pad|>": 1065}
    # Reload both through load(), as the release does from checkpoint metadata.
    t2 = RegexTokenizer(GPT4_SPLIT_PATTERN)
    t2.load(io.BytesIO(model_text(text).encode("utf-8")))
    s2 = CodebookTokenizer(GPT4_SPLIT_PATTERN)
    s2.load(io.BytesIO(model_text(speech).encode("utf-8")))
    texts = ["<|startoftext|>The quick brown fox.<|endoftext|>", "<|startoftext|>It's 12345 o'clock, isn't it?<|endoftext|>",
             "<|startoftext|>Hello   there,\nnew line\tand tabs!!<|endoftext|>", "<|startoftext|>Ünïcödé ½ 漢字 don't<|endoftext|>"]
    code_cases = [codes[:50].tolist(), codes[150:260].tolist(), [3, 3, 3, 3, 7, 1]]
    out["tokenizers"] = {
        "text_model": model_text(text), "speech_model": model_text(speech),
        "text_vocab": len(t2.vocab), "speech_vocab": len(s2.vocab),
        "texts": [{"text": s, "ids": t2.encode(s, allowed_special="all")} for s in texts],
        "codes": [{"codes": c, "ids": s2.encode(" ".join(str(x) for x in c))} for c in code_cases],
        "decode": [{"ids": ids, "codes": [x for x in s2.decode_int(ids) if isinstance(x, int)]}
                   for ids in [s2.encode(" ".join(str(x) for x in code_cases[1])) + [1064], [5, 1030, 1064, 7]]],
    }


def networks(out):
    torch.manual_seed(11)
    ar = CodecLM(n_vocab=40, dim=32, nhead=4, n_layers=2, n_spk_layers=1, dim_ff_scale=7 / 3, sliding_window=3000).double().eval()
    with torch.no_grad():
        ar.pos_embedding.alpha.fill_(0.7)
    ref = torch.randint(0, 1024, (1, 9, 8))
    ref[0, 7:, 0] = 1024  # padding from frame 7 on
    tokens = torch.randint(0, 40, (1, 12))
    with torch.no_grad():
        logits = ar(tokens, None, spk_reference=ref)
        spk = ar.get_spk_embedding(ref)
    out["ar"] = {"state": {k: tolist(v) for k, v in ar.state_dict().items()}, "reference": ref[0].tolist(), "tokens": tokens[0].tolist(),
                 "logits": tolist(logits[0]), "speaker": tolist(spk[0])}

    nar = ResidualTransformer(n_text_vocab=30, n_quant=1025, dim=32, nhead=4, enc_layers=2, dec_layers=2, n_spk_layers=1,
                              t_emb_dim=16, p_cond_drop=0, dropout=0).double().eval()
    with torch.no_grad():
        for name in ["cond_pos_embedding", "pos_embedding", "ref_pos_embedding"]:
            getattr(nar, name).alpha.fill_(0.5 + 0.1 * len(name) % 3)
    c_text = torch.randint(0, 30, (1, 7))
    c_codes = torch.randint(0, 1024, (1, 6, 8))
    x = torch.randint(0, 1025, (1, 5, 8))
    cases = []
    for t, drop in [(0, False), (37, False), (150, True)]:
        with torch.no_grad():
            o = nar(c_text, c_codes.clone(), torch.tensor([7]), torch.tensor([6]), x, torch.zeros(1, 5, dtype=torch.bool),
                    torch.tensor([t]), drop_cond=drop)
        cases.append({"t": t, "drop": drop, "logits": tolist(o[0].permute(0, 2, 1))})  # [L, n_quant, K]
    out["nar"] = {"state": {k: tolist(v) for k, v in nar.state_dict().items()}, "text": c_text[0].tolist(), "reference": c_codes[0].tolist(),
                  "x": x[0].tolist(), "cases": cases}
    return nar, c_text, c_codes, x


def scaled_intent(self, log_x_t, t, c, jump_len):
    """q_pred_one_timestep_scaled with its ramp along the sequence, as its variables intend: the release broadcasts the
    [1, L, 1] ramp onto the codebook axis (it fails unless L is 8), a bug the port fixes (the user's decision)."""
    dt = log_x_t.dtype
    log_alpha_t = diffuser.extract(self.log_alpha, t, log_x_t.shape).to(dt)
    log_1_min_alpha_t = diffuser.extract(self.log_1_min_alpha, t, log_x_t.shape).to(dt)
    xax = torch.arange(0, log_x_t.shape[1], 1).to(log_x_t.device)
    aa = log_x_t.shape[1] * (c / jump_len)
    sig = 1 / (1 + torch.exp(-(xax - aa + 20) / 8))
    log_alpha_t = (torch.log(1 / sig)[None, :, None, None] + log_alpha_t).clamp(-torch.inf, 0)
    log_1_min_alpha_t = torch.log(sig)[None, :, None, None] + log_1_min_alpha_t
    return diffuser.log_add_exp(log_x_t + log_alpha_t, log_1_min_alpha_t - math.log(self.num_classes))


def diffusion(out, nar, c_text, c_codes, x):
    MultinomialDiffusion.q_pred_one_timestep_scaled = scaled_intent
    torch.set_default_dtype(torch.float64)
    diff = MultinomialDiffusion(1025, timesteps=8, dtype=torch.float32)  # the release's default: float32 tables
    entry = {"timesteps": 8, "log_alpha": tolist(diff.log_alpha), "log_1_min_alpha": tolist(diff.log_1_min_alpha),
             "log_cumprod_alpha": tolist(diff.log_cumprod_alpha), "log_1_min_cumprod_alpha": tolist(diff.log_1_min_cumprod_alpha),
             "schedules": [{"T": T, "jl": jl, "js": js, "ts": get_schedule(T, jump_len=jl, jump_n_sample=js)}
                           for T, jl, js in [(8, 1, 1), (10, 2, 3), (12, 3, 2)]]}
    g = torch.Generator().manual_seed(5)
    x0 = torch.randint(0, 1025, (1, 4, 8), generator=g)
    xt = torch.randint(0, 1025, (1, 4, 8), generator=g)
    pred = torch.randn(1, 4, 8, 1025, generator=g, dtype=torch.float64)
    log_x0 = diffuser.index_to_log_onehot(x0, 1025, dtype=torch.float64)
    log_xt = diffuser.index_to_log_onehot(xt, 1025, dtype=torch.float64)
    log_pred = F.log_softmax(pred, dim=-1)
    losses = []
    for t in [0, 3, 7]:
        tt = torch.tensor([t])
        post = diff.q_posterior(log_x0, log_xt, tt)
        losses.append({"t": t, "posterior_head": tolist(post[0, 0, 0, :12]), "loss": diff.compute_Lt(log_x0, log_xt, log_pred, tt).item(),
                       "loss_no_prior": diff.compute_Lt(log_x0, log_xt, log_pred, tt, include_kl_prior=False).item()})
    entry.update(x0=x0[0].tolist(), xt=xt[0].tolist(), pred=tolist(pred), losses=losses)
    scaled = diff.q_pred_one_timestep_scaled(log_xt, torch.tensor([2]), 0.6, 1)
    entry["scaled_head"] = tolist(scaled[0, :, :, :6])

    # perform_simple_inference with torch.rand_like / torch.randint replaced by closed-form patterns.
    state = {"u": 0, "r": 0}

    def rand_like(t, **kw):
        n = t.numel()
        idx = torch.arange(state["u"], state["u"] + n, dtype=torch.float64)
        state["u"] += n
        return ((torch.sin(idx * 12.9898) * 43758.5453) % 1.0).abs().reshape(t.shape).to(t.dtype)

    def randint(low, high, size, dtype=None, device=None, **kw):
        n = int(torch.tensor(size).prod())
        idx = torch.arange(state["r"], state["r"] + n)
        state["r"] += n
        return ((idx * 7919 + 13) % high).reshape(size).to(dtype or torch.long)

    # The inference cases test the diffusion logic: a copy of the NAR without its float32 inputs (the sine tables' α and
    # the timestep embedding's first layer zeroed), whose logits are then exact in float64, so no near-tie flips a sample.
    # (Changed in place: these modules do not survive copy.deepcopy, as the release notes; the network tests ran already.)
    with torch.no_grad():
        for name in ["cond_pos_embedding", "pos_embedding", "ref_pos_embedding"]:
            getattr(nar, name).alpha.zero_()
        nar.timestep_encoder_emb[0].weight.zero_()
        nar.timestep_decoder_emb[0].weight.zero_()
    entry["inference_state"] = {k: tolist(v) for k, v in nar.state_dict().items()}
    saved = (torch.rand_like, torch.randint)
    torch.rand_like, torch.randint = rand_like, randint
    infer = []
    try:
        for deep, jl, js, guidance in [(False, 1, 1, 3.0), (True, 1, 1, 3.0), (True, 2, 2, 1.0)]:
            state["u"] = state["r"] = 0
            quant0 = torch.tensor([[5, 9, 1, 1023, 77]])
            batch = (c_text, c_codes.clone(), torch.tensor([7]), torch.tensor([6]), quant0[..., None].repeat(1, 1, 8),
                     torch.zeros(1, 5, dtype=torch.bool))
            dsh = DSH(last_greedy=True, x_0_temp=0.7, guidance_w=guidance, deep_clone=deep, jump_len=jl, jump_n_sample=js,
                      q0_override_steps=2, enable_kevin_scaled_inference=True, progress=False)
            d = MultinomialDiffusion(1025, timesteps=6, dtype=torch.float32)
            with torch.no_grad():
                res = perform_simple_inference(nar, batch, d, d.num_timesteps, torch.float64, dsh=dsh, retain_quant0=True)
            infer.append({"deep": deep, "jump_len": jl, "jump_n_sample": js, "guidance": guidance, "timesteps": 6, "q0_override": 2,
                          "quant0": quant0[0].tolist(), "codes": res[0].tolist()})
    finally:
        torch.rand_like, torch.randint = saved
    entry["inference"] = infer
    out["diffusion"] = entry
    torch.set_default_dtype(torch.float32)


def samplers(out):
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(1, 50, generator=g, dtype=torch.float64) * 3
    previous = torch.tensor([[3, 3, 7, 9, 3, 12, 7, 40, 41, 3]])
    a = freq_rep_penalty(logits.clone(), previous, 3, 0.4, 6)
    b = early_eos_penalty(a.clone(), 4, 10, 0.5, 1.0, eos_index=12)
    c = top_k_top_p_filtering(b.clone() / 0.7, top_k=20, top_p=0.6)
    d = apply_typical_p(b.clone(), 0.5)
    inf = lambda t: [None if math.isinf(v) else v for v in tolist(t)]
    out["samplers"] = {"logits": tolist(logits), "previous": previous[0].tolist(), "penalized": inf(a), "eos": inf(b), "topkp": inf(c), "typical": inf(d)}


def trimming(out):
    g = torch.Generator().manual_seed(2)
    n = 24000
    t = torch.arange(n, dtype=torch.float64) / 24000
    y = 0.001 * torch.randn(n, generator=g, dtype=torch.float64)
    y[6000:17000] += 0.5 * torch.sin(2 * math.pi * 300 * t[6000:17000])
    yt, idx = trim(y, top_db=27)
    out["trim"] = {"audio": tolist(y), "start": int(idx[0]), "end": int(idx[1])}


def main():
    out = {}
    tokenizers(out)
    nar, c_text, c_codes, x = networks(out)
    diffusion(out, nar, c_text, c_codes, x)
    samplers(out)
    trimming(out)
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "mars5_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle, ensure_ascii=False)
        handle.write("\n")
    print("wrote", os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
