"""CosyVoice 2 language model reference: FunAudioLLM/CosyVoice's Qwen2LM (cosyvoice/llm/llm.py) over a Hugging Face
Qwen2ForCausalLM (transformers 5.18.0). Pass checkouts with --cosyvoice and --matcha (cosyvoice imports Matcha-TTS on
the way).

Builds a tiny model whose weights follow an exact integer formula of each parameter's name and index (weights(),
which the C# test repeats, so no weights are stored) and runs it in float64, recording in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice2_lm_reference.json: the log-probabilities of the unistream
sequence, and Qwen2LM.forward's loss for three utterances, the bistream/unistream draw (random.random) pinned: the
unistream form, the bistream form with a short last text chunk, and the bistream form whose text fills its chunks
exactly (the last chunk then holds no text); and the unistream loss after one SGD step (lr 1e-3). Hugging Face builds
the rotary table, the RMSNorm and the attention softmax in float32 whatever the model's dtype; here they are float64,
as the C# port computes them. Default:
verify; --write: regenerate.
"""
import argparse
import base64
import random
import types
from unittest import mock

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture
from cosyvoice_flow_reference import stub_imports

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice2_lm_reference.json"
# 896 wide: prepare_lm_input_target hard-codes an empty [0, 896] instruct block.
QWEN = dict(vocab_size=40, hidden_size=896, num_hidden_layers=2, num_attention_heads=14, num_key_value_heads=2,
            intermediate_size=32, rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=128, tie_word_embeddings=True)
SPEECH_TOKENS = 30
CASES = [
    ("unistream", 0.9, [3, 17, 8, 1, 11, 25, 6], [4, 9, 0, 7, 2, 11, 29, 13]),
    ("bistream", 0.1, [3, 17, 8, 1, 11, 25, 6], [(5 * i + 3) % 30 for i in range(25)]),
    ("bistream_exact", 0.1, [3, 17, 8, 1, 11], [(7 * i + 1) % 30 for i in range(16)]),
]
LEARNING_RATE = 1e-3


def float64_rotary(self, x, position_ids):
    inv = (1.0 / (QWEN["rope_theta"] ** (torch.arange(0, self.inv_freq.numel() * 2, 2, dtype=torch.float64)
                                         / (self.inv_freq.numel() * 2))))
    freqs = position_ids[:, :, None].to(torch.float64) * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def float64_rms_norm(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def float64_softmax(softmax):
    # eager_attention_forward softmaxes with dtype=torch.float32 whatever the model's dtype.
    def patched(x, dim=None, dtype=None, **kwargs):
        return softmax(x, dim=dim, **kwargs)
    return patched


def build(state=None):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    from cosyvoice.llm.llm import Qwen2Encoder, Qwen2LM
    modeling_qwen2.Qwen2RotaryEmbedding.forward = float64_rotary
    modeling_qwen2.Qwen2RMSNorm.forward = float64_rms_norm
    if not getattr(modeling_qwen2.nn.functional.softmax, "_float64", False):
        patched = float64_softmax(modeling_qwen2.nn.functional.softmax)
        patched._float64 = True
        modeling_qwen2.nn.functional.softmax = patched
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2294)
    config = Qwen2Config(**QWEN)
    config._attn_implementation = "eager"
    encoder = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = Qwen2ForCausalLM(config)
    model = Qwen2LM(llm_input_size=QWEN["hidden_size"], llm_output_size=QWEN["hidden_size"], speech_token_size=SPEECH_TOKENS,
                    llm=encoder, sampling=None, length_normalized_loss=True, lsm_weight=0, mix_ratio=[5, 15]).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.tensor(weights(name, p.numel()), dtype=torch.float64).reshape(p.shape))
    return model


def weights(name, count):
    # Exact in float64 and computed the same way by the C# test: a value per index from integer arithmetic on the
    # parameter's name, around 1 for norm weights and around 0 otherwise.
    h = sum((j + 1) * ord(ch) for j, ch in enumerate(name)) % 100003
    base = 1.0 if "norm" in name else 0.0
    return [base + 0.1 * (((i * 7919 + h) % 2003) / 2003.0 - 0.5) for i in range(count)]


def loss(model, text, speech, draw):
    batch = {"text_token": torch.tensor([text]), "text_token_len": torch.tensor([len(text)]),
             "speech_token": torch.tensor([speech]), "speech_token_len": torch.tensor([len(speech)])}
    with mock.patch.object(random, "random", return_value=draw):
        return model(batch, torch.device("cpu"))["loss"]


def run(model):
    name, draw, text, speech = CASES[0]
    with torch.no_grad():
        sos = model.llm_embedding.weight[0].reshape(1, 1, -1)
        task = model.llm_embedding.weight[1].reshape(1, 1, -1)
        emb = model.llm.model.model.embed_tokens(torch.tensor([text]))
        lm_input = torch.concat([sos, emb, task, model.speech_embedding(torch.tensor([speech]))], dim=1)
        out, _ = model.llm(lm_input, torch.tensor([lm_input.size(1)]))
        logp = model.llm_decoder(out).log_softmax(dim=-1)
    losses = {n: float(loss(model, t, s, d).detach()) for n, d, t, s in CASES}
    first = loss(model, text, speech, draw)
    parameters = [p for p in model.parameters()]
    grads = torch.autograd.grad(first, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
        after = float(loss(model, text, speech, draw).detach())
    return logp, losses, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stub_imports(args.cosyvoice, args.matcha)
    if args.write:
        model = build()
        logp, losses, after = run(model)
        data = {"qwen": QWEN, "speech_tokens": SPEECH_TOKENS,
                "cases": [{"name": n, "text": t, "speech": s} for n, _, t, s in CASES], "learning_rate": LEARNING_RATE,
                "parameters": [[n, list(p.shape)] for n, p in build().named_parameters()],
                "logp": [float(v) for v in logp.reshape(-1).tolist()], "losses": losses, "loss_after_sgd": after}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build()
    logp, losses, after = run(model)
    check = _fixture.Comparison("CosyVoice 2 LM")
    check.floats("logp", data["logp"], logp.reshape(-1).tolist(), 1e-12, float)
    for name in losses:
        check.floats(name, [data["losses"][name]], [losses[name]], 1e-12, float)
    check.floats("loss_after_sgd", [data["loss_after_sgd"]], [after], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
