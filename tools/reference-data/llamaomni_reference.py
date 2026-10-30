"""LLaMA-Omni reference: the release's speech adaptor (omni_speech/model/speech_projector/speech_projector.py
EncoderProjectorConcat) and CTC speech decoder (speech_generator/speech_generator.py SpeechGeneratorCTC). Pass the
ictnlp/LLaMA-Omni checkout with --omni.

The release runs SpeechGeneratorCTC's LlamaDecoderLayers under transformers 4.43 with flash attention, handing each layer
the padding mask and position ids. transformers 4.57's layer takes the rotary embeddings and a 4-D mask instead, so each
layer's call is adapted, computing them as 4.43 did: the rotary table of the same config (Llama 3 scaling included), a
causal mask (flash attention's is_causal) over the valid frames. Everything else is the release's code, run in float64;
the CTC head's log-softmax, the rotary table, RMSNorm and the attention softmax, which compute in float32, are made float64.

Records, in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/llamaomni_reference.json, for a tiny adaptor and decoder
(seeded):
  - the adaptor's output for 23 encoder frames (k = 5: the last 3 discarded);
  - the decoder's CTC loss for two utterances' hidden states and units (forward with labels), its prediction (argmax with
    the padding blank) for one utterance, and the loss after one SGD step (lr 0.1) on the decoder;
with --shard (the release's model-00004-of-00004.safetensors), in llamaomni_realweights_reference.json, the released
adaptor on 50 encoder frames and decoder on 6 hidden states. Default: verify; --write: regenerate.
"""
import argparse
import base64
import importlib.util
import os
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/llamaomni_reference.json"
REAL_FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/llamaomni_realweights_reference.json"
IGNORE = -100


def load_release(root):
    omni = types.ModuleType("omni_speech")
    omni.__path__ = []
    sys.modules["omni_speech"] = omni
    constants = types.ModuleType("omni_speech.constants")
    constants.IGNORE_INDEX = IGNORE
    sys.modules["omni_speech.constants"] = constants

    def load(name, path, float64=False):
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        if float64:
            # The CTC head's log-softmax casts to float32 (logits.float(), dtype=torch.float32); kept in float64 here.
            assert source.count("ctc_logits.float(), dim=-1, dtype=torch.float32") == 2
            source = source.replace("ctc_logits.float(), dim=-1, dtype=torch.float32", "ctc_logits, dim=-1, dtype=torch.float64")
        module = types.ModuleType(name)
        module.__file__ = path
        exec(compile(source, path, "exec"), module.__dict__)
        return module
    projector = load("speech_projector", os.path.join(root, "omni_speech/model/speech_projector/speech_projector.py"))
    generator = load("speech_generator", os.path.join(root, "omni_speech/model/speech_generator/speech_generator.py"), float64=True)
    return projector.EncoderProjectorConcat, generator.SpeechGeneratorCTC


def adapt_layers(model, config):
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    # LlamaRMSNorm computes in float32 (hidden_states.to(torch.float32)); the same formula in the input's precision.
    def rms_norm(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))
    modeling_llama.LlamaRMSNorm.forward = rms_norm
    # Eager attention's softmax also runs in float32 (softmax(..., dtype=torch.float32)); kept in the scores' precision.
    softmax = torch.nn.functional.softmax
    modeling_llama.nn.functional.softmax = lambda x, dim=None, dtype=None, **k: softmax(x, dim=dim)
    _config = type(config)(**config.to_dict())
    n_layers, n_dims, n_heads, n_inter = map(int, config.ctc_decoder_config[1:-1].split(","))
    _config.hidden_size, _config.num_attention_heads, _config.num_key_value_heads = n_dims, n_heads, n_heads
    # HF's rotary table is float32 (inv_freq and the angles both .float()); made float64 by redirecting Tensor.float.
    original_float = torch.Tensor.float

    def float64(build):
        torch.Tensor.float = lambda self, *a, **k: self.double()
        try:
            return build()
        finally:
            torch.Tensor.float = original_float
    rotary = float64(lambda: LlamaRotaryEmbedding(_config))
    for layer in model.layers:
        layer.self_attn.config._attn_implementation = "eager"
        original = layer.forward

        def forward(hidden_states, attention_mask=None, position_ids=None, original=original, **kwargs):
            length = hidden_states.shape[1]
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
            allowed = causal[None] & attention_mask[:, None, :]
            bias = torch.zeros(allowed.shape, dtype=hidden_states.dtype).masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)[:, None]
            cos, sin = float64(lambda: rotary(hidden_states, position_ids))
            out = original(hidden_states, attention_mask=bias, position_ids=position_ids, position_embeddings=(cos, sin))
            return out if isinstance(out, tuple) else (out,)
        layer.forward = forward


def tiny_config():
    from transformers import LlamaConfig
    config = LlamaConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=2, intermediate_size=32, rms_norm_eps=1e-5,
                         rope_theta=500000.0, rope_scaling={"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                                           "original_max_position_embeddings": 32, "rope_type": "llama3"},
                         max_position_embeddings=256)
    config.ctc_decoder_config = "(2,8,2,16)"
    # transformers 4.43 (the release's) sizes a head as hidden_size / heads of the decoder's own width; 4.57 reads
    # config.head_dim, which the copied config would carry over from the 16-wide language model.
    config.head_dim = 8 // 2
    config.ctc_upsample_factor = 3
    config.unit_vocab_size = 12
    config.speech_encoder_ds_rate = 5
    config.speech_encoder_hidden_size = 6
    return config


def run_generator(generator, reps, labels, units, single):
    loss = generator(reps, labels, units)
    with torch.no_grad():
        prediction = generator.predict(single)
    return loss, prediction


def floats(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni", required=True)
    parser.add_argument("--shard")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    Projector, Generator = load_release(args.omni)
    torch.set_default_dtype(torch.float64)

    if args.shard:
        from transformers import LlamaConfig
        from safetensors import safe_open
        config = LlamaConfig(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, intermediate_size=14336, rms_norm_eps=1e-5,
                             rope_theta=500000.0, rope_scaling={"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                                                               "original_max_position_embeddings": 8192, "rope_type": "llama3"},
                             max_position_embeddings=131072)
        config.ctc_decoder_config, config.ctc_upsample_factor, config.unit_vocab_size = "(2,4096,32,11008)", 25, 1000
        config.speech_encoder_ds_rate, config.speech_encoder_hidden_size = 5, 1280
        projector, generator = Projector(config), Generator(config)
        with safe_open(args.shard, "pt") as f:
            projector.load_state_dict({k[len("model.speech_projector."):]: f.get_tensor(k).double() for k in f.keys() if k.startswith("model.speech_projector.")})
            generator.load_state_dict({k[len("speech_generator."):]: f.get_tensor(k).double() for k in f.keys() if k.startswith("speech_generator.")})
        projector, generator = projector.double().eval(), generator.double().eval()
        adapt_layers(generator, config)
        stored = None if args.write else _fixture.load(REAL_FIXTURE)
        rng = torch.Generator().manual_seed(9)
        encoder = torch.tensor(stored["encoder"]).reshape(1, 50, 1280) if stored else torch.randn(1, 50, 1280, generator=rng)
        reps = torch.tensor(stored["reps"]).reshape(6, 4096) if stored else 2 * torch.randn(6, 4096, generator=rng)
        units = stored["units"] if stored else [(53 * i + 7) % 1000 for i in range(40)]
        with torch.no_grad():
            adapted = projector(encoder)
            loss = generator([reps], torch.zeros(1, 6, dtype=torch.long), torch.tensor([units]))
            prediction = generator.predict(reps)
        results = {"adapted": floats(adapted), "loss": float(loss), "prediction": prediction[0].tolist()}
        if args.write:
            _fixture.save(REAL_FIXTURE, {"encoder": floats(encoder), "reps": floats(reps), "units": units, **results}, __file__)
            print("wrote", REAL_FIXTURE, results["loss"], results["prediction"][:20])
            return
        check = _fixture.Comparison("LLaMA-Omni, released adaptor and decoder")
        check.floats("adapted", stored["adapted"], results["adapted"], 1e-12, float)
        check.floats("loss", [stored["loss"]], [results["loss"]], 1e-12, float)
        check.exact("prediction", stored["prediction"], results["prediction"])
        check.report(exact=False)
        return

    stored = None if args.write else _fixture.load(FIXTURE)
    config = tiny_config()
    torch.manual_seed(2294)
    projector, generator = Projector(config), Generator(config)
    with torch.no_grad():
        for module in (projector, generator):
            for name, p in module.named_parameters():
                if "layernorm" in name:
                    p.add_(0.2 * torch.randn_like(p))
                p.copy_(p.float().double())
    if stored:
        projector.load_state_dict(load_safetensors(base64.b64decode(stored["projector_base64"])))
        generator.load_state_dict(load_safetensors(base64.b64decode(stored["generator_base64"])))
    projector, generator = projector.double().eval(), generator.double().eval()
    initial = {k: v.detach().float().contiguous().clone() for k, v in generator.state_dict().items()}
    adapt_layers(generator, config)
    rng = torch.Generator().manual_seed(4)
    encoder = torch.tensor(stored["encoder"]).reshape(1, 23, 6) if stored else torch.randn(1, 23, 6, generator=rng)
    # Two utterances: hidden states over a prompt and a response, the labels marking the response (IGNORE elsewhere).
    reps = [torch.tensor(stored["reps0"]).reshape(7, 16), torch.tensor(stored["reps1"]).reshape(5, 16)] if stored else \
        [torch.randn(7, 16, generator=rng), torch.randn(5, 16, generator=rng)]
    labels = torch.tensor([[IGNORE, IGNORE, 3, 9, 1, 2, 5], [IGNORE, 4, 4, 8, 6, IGNORE, IGNORE]])
    units = torch.tensor([[3, 3, 7, 1, 11, 2, 2, 9, 4, 10, 0, 5], [6, 6, 6, 2, 8, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE]])
    reps_padded = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)
    with torch.no_grad():
        adapted = projector(encoder)
    loss, prediction = run_generator(generator, reps_padded, labels, units, reps[0])
    grads = torch.autograd.grad(loss, list(generator.parameters()))
    with torch.no_grad():
        for p, g in zip(generator.parameters(), grads):
            p -= 0.1 * g
    after, _ = run_generator(generator, reps_padded, labels, units, reps[0])
    results = {"adapted": floats(adapted), "loss": float(loss), "prediction": prediction[0].tolist(), "loss_after_sgd": float(after)}
    if args.write:
        state = lambda m: base64.b64encode(save_safetensors({k: v.detach().float().contiguous() for k, v in m.state_dict().items()})).decode("ascii")
        _fixture.save(FIXTURE, {"encoder": floats(encoder), "reps0": floats(reps[0]), "reps1": floats(reps[1]), "labels": labels.tolist(),
                                "units": units.tolist(), "projector_base64": state(projector), "generator_base64": base64.b64encode(save_safetensors(initial)).decode("ascii"),
                                **results}, __file__)
        print("wrote", FIXTURE, results["loss"], results["prediction"])
        return
    check = _fixture.Comparison("LLaMA-Omni adaptor and CTC decoder")
    for k in ("adapted",):
        check.floats(k, stored[k], results[k], 1e-12, float)
    for k in ("loss", "loss_after_sgd"):
        check.floats(k, [stored[k]], [results[k]], 1e-12, float)
    check.exact("prediction", stored["prediction"], results["prediction"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
