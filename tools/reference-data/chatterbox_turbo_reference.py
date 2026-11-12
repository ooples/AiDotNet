"""Chatterbox Turbo/Nano reference (resemble-ai/chatterbox 5de7a54): T3 with a GPT-2 backbone (tts_turbo's T3Config:
no learned text/speech positions, no perceiver, no emotion input, a biased speech head), S3Gen's mean-flow
ConditionalDecoder (meanflow=True, its IntMeanFlow time mixer) and ConditionalCFM.basic_euler, tts_turbo.punc_norm and
the release's GPT-2 tokenizer (AutoTokenizer over ResembleAI/chatterbox-turbo).

Records, in float64 with seeded weights: the tiny T3's text and speech logits, its embeddings, the next-token loss and the
logits after one SGD step; the tiny causal mean-flow decoder's field for one (t, r) and two Euler steps of basic_euler;
punc_norm of some strings; the tokenizer's ids for some sentences.
Usage: python tools/reference-data/chatterbox_turbo_reference.py --chatterbox <checkout> --turbo <chatterbox-turbo dir>
       [--write]   (in .venv-chatterbox)
"""
import argparse
import base64
import math
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chatterbox_turbo_reference.json"
MEL = 4
DECODER = dict(in_channels=4 * MEL, out_channels=MEL, causal=True, channels=[8], dropout=0.0, attention_head_dim=4, n_blocks=2,
               num_mid_blocks=2, num_heads=2, act_fn="gelu", meanflow=True)
FRAMES = 8
SENTENCES = ["Hello there, this is Chatterbox Turbo.", "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store.",
             "Numbers 123 and   spaces; also - dashes...", "[laugh] [sigh] mixed [cough]tokens"]
PUNC = ["", "hello world", "Done…", "a: b; c — d", "it’s “quoted”", "ok,", "trailing   "]


def import_chatterbox(checkout):
    root = Path(checkout).resolve() / "src"
    sys.path.insert(0, str(root))


def t3_reference():
    from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
    from chatterbox.models.t3.modules.t3_config import T3Config
    import chatterbox.models.t3.modules.cond_enc as conditioning
    from chatterbox.models.t3.t3 import T3

    torch.manual_seed(5302)
    LLAMA_CONFIGS["audit_gpt"] = dict(activation_function="gelu_new", attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
                                      layer_norm_epsilon=1e-5, model_type="gpt2", n_ctx=64, n_embd=16, hidden_size=16,
                                      n_head=2, n_layer=2, n_positions=64, vocab_size=8, initializer_range=0.02)
    hp = T3Config(text_tokens_dict_size=32)
    hp.llama_config_name = "audit_gpt"
    hp.start_text_token = 30
    hp.stop_text_token = 31
    hp.start_speech_token = 18
    hp.stop_speech_token = 19
    hp.speech_tokens_dict_size = 20
    hp.speaker_embed_size = 6
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 3
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    model = T3(hp)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "ln_" in name:
                p.add_(0.1 * torch.randn_like(p))
    del model.tfmr.wte
    model = model.double().eval()
    speaker = torch.randn(1, 6, dtype=torch.float64)
    prompt = torch.tensor([[2, 4, 6]])
    text = torch.tensor([[30, 1, 3, 31]])
    speech = torch.tensor([[18, 4, 2, 5, 19]])

    def condition():
        return conditioning.T3Cond(speaker_emb=speaker, cond_prompt_speech_tokens=prompt)

    arguments = dict(text_tokens=text, text_token_lens=torch.tensor([text.shape[1]]), speech_tokens=speech,
                     speech_token_lens=torch.tensor([speech.shape[1]]))
    output = model(t3_cond=condition(), **arguments)
    embeddings, _ = model.prepare_input_embeds(t3_cond=condition(), text_tokens=text, speech_tokens=speech, cfg_weight=0.0)
    loss = F.cross_entropy(output.text_logits[:, :-1].reshape(-1, 32), text[:, 1:].reshape(-1))
    loss = loss + F.cross_entropy(output.speech_logits[:, :-1].reshape(-1, 20), speech[:, 1:].reshape(-1))
    flat = lambda tensor: tensor.detach().flatten().tolist()
    data = dict(weights={k: flat(v) for k, v in model.state_dict().items()},
                weight_shapes={k: list(v.shape) for k, v in model.state_dict().items()},
                speaker=flat(speaker), prompt=flat(prompt), text=flat(text), speech=flat(speech),
                text_logits=flat(output.text_logits), speech_logits=flat(output.speech_logits), embeddings=flat(embeddings),
                next_token_loss=loss.item())
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.sub_(0.01 * p.grad)
    after = model(t3_cond=condition(), **arguments)
    data.update(after_sgd_text=flat(after.text_logits), after_sgd_speech=flat(after.speech_logits))
    return data


def decoder_reference():
    import chatterbox.models.s3gen.decoder as decoder
    import chatterbox.models.s3gen.matcha.decoder as matcha_decoder

    def mask_to_bias(mask, dtype):
        assert mask.dtype == torch.bool
        return (1.0 - mask.to(dtype)) * -1.0e+10
    decoder.mask_to_bias = mask_to_bias

    def sinusoidal(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        step = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float64) * -step)
        emb = scale * x.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    matcha_decoder.SinusoidalPosEmb.forward = sinusoidal
    decoder.SinusoidalPosEmb.forward = sinusoidal

    torch.manual_seed(2511)
    model = decoder.ConditionalDecoder(**DECODER)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name or name.endswith("block.1.weight") or "time_embed_mixer" in name:
                p.add_(0.1 * torch.randn_like(p))
    model = model.double().eval()
    x = torch.randn(1, MEL, FRAMES, dtype=torch.float64)
    mu = torch.randn(1, MEL, FRAMES, dtype=torch.float64)
    spks = torch.randn(1, MEL, dtype=torch.float64)
    cond = torch.randn(1, MEL, FRAMES, dtype=torch.float64)
    mask = torch.ones(1, 1, FRAMES, dtype=torch.float64)
    t = torch.tensor([0.3], dtype=torch.float64)
    r = torch.tensor([0.8], dtype=torch.float64)
    with torch.no_grad():
        field = model(x, mask, mu, t, spks=spks, cond=cond, r=r)
        # ConditionalCFM.basic_euler over torch.linspace(0, 1, 3).
        span = torch.linspace(0, 1, 3, dtype=torch.float64)
        y = x.clone()
        for a, b in zip(span[:-1], span[1:]):
            y = y + (b - a) * model(y, mask, mu, a[None], spks=spks, cond=cond, r=b[None])
    state = {k: v.detach().to(torch.float32).contiguous() for k, v in model.state_dict().items()}
    flat = lambda tensor: tensor.detach().flatten().tolist()
    return dict(config={k: v for k, v in DECODER.items()}, mel=MEL, frames=FRAMES, t=0.3, r=0.8, x=flat(x), mu=flat(mu),
                spks=flat(spks), cond=flat(cond), field=flat(field), sample=flat(y),
                state={k: flat(v.double()) for k, v in model.state_dict().items()},
                state_shapes={k: list(v.shape) for k, v in model.state_dict().items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chatterbox", required=True)
    parser.add_argument("--turbo", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import_chatterbox(args.chatterbox)
    torch.set_default_dtype(torch.float64)
    from chatterbox.tts_turbo import punc_norm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.turbo)
    data = dict(t3=t3_reference(), decoder=decoder_reference(),
                punc_norm=[dict(text=t, normalized=punc_norm(t)) for t in PUNC],
                tokens=[dict(text=t, ids=tokenizer(punc_norm(t)).input_ids) for t in SENTENCES])
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "loss", data["t3"]["next_token_loss"])
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Chatterbox Turbo")
    check.exact("tokens", stored["tokens"], data["tokens"])
    check.exact("punc_norm", stored["punc_norm"], data["punc_norm"])
    check.floats("speech logits", stored["t3"]["speech_logits"], data["t3"]["speech_logits"], 1e-12, float)
    check.floats("decoder field", stored["decoder"]["field"], data["decoder"]["field"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
