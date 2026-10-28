"""Reference fixture for Dia (Nari Labs 2025): nari-labs/dia at 876125e (dia.model.Dia over dia.layers.DiaModel, as the
release defines them) at a tiny size, and transformers' DiaForConditionalGeneration (4.57.1, the port written with Nari
Labs and the only training code) on the same weights, in float64.

The tiny model: an encoder of 2 layers (width 16, 2 heads of 8, feed-forward 24) over the 256 byte tokens, and a decoder of
2 layers (width 16, 4 query heads and 2 key-value heads of 8, cross-attention of 2 heads of 8, feed-forward 24) over 3
channels of 20 tokens (codes 0-15, EOS 16, PAD 17, BOS 18), delay pattern [0, 2, 3]. Every parameter is a closed form of
its release name and flat index i: s = sin(0.37 i + 0.11 len(name) + 0.0013 (i^2 mod 9973)); RMSNorm weights are 1 + 0.2 s,
the embeddings 0.5 s, every other kernel 0.5 s / sqrt(fan-in) (its first dimension, or the first two for o_proj).

The release computes RoPE, its timescale and the decoder's logits in float32, and torch's CPU scaled_dot_product_attention
is replaced by the same arithmetic; transformers computes its inverse frequencies, rotary embedding, RMSNorm and eager softmax in float32; each is
replaced by the same arithmetic in float64.

Records: the encoder's output for the text (its conditional branch); a generation through Dia.generate with an audio prompt
of 4 frames, greedy (temperature 0) with CFG scale 3 and a CFG filter of top-k 5 until the length limit: every step's
conditional and unconditional logits and the codes it returns; and transformers' loss for the text with 6 frames of codes,
the decoder inputs and labels built as DiaProcessor builds them for training.

Run with ~/source/repos/.venv-dia (torch 2.6.0 cpu, transformers 4.57.1, pydantic) and the dia clone in refsrc/dia.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/dia_reference.json.
"""
import inspect
import json
import math
import os
import sys
import textwrap

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "dia"))
torch.set_default_dtype(torch.float64)

import torch.nn.functional as F  # noqa: E402
import dia.layers as dia_layers  # noqa: E402
import dia.model as dia_model  # noqa: E402
from dia.config import DecoderConfig, DiaConfig, EncoderConfig  # noqa: E402
import transformers.models.dia.modeling_dia as hf_dia  # noqa: E402
from transformers import DiaConfig as HfDiaConfig, DiaForConditionalGeneration  # noqa: E402
from transformers.models.dia.configuration_dia import DiaDecoderConfig, DiaEncoderConfig  # noqa: E402
from transformers.models.dia.processing_dia import DiaProcessor  # noqa: E402

SDPA = F.scaled_dot_product_attention
TEXT = "[S1] Dia speaks. [S2] Yes!"
CHANNELS, VOCAB, EOS, PAD, BOS, DELAYS = 3, 20, 16, 17, 18, [0, 2, 3]


def attention64(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if q.dtype != torch.float64:
        return SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) * (scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
    if is_causal:
        attn_mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask
    return torch.softmax(scores, dim=-1) @ v


F.scaled_dot_product_attention = attention64


def timescale64(module):
    half = module.embedding_dims // 2
    fraction = (2.0 * torch.arange(0, half, dtype=torch.float64)) / module.embedding_dims
    module.timescale = module.min_timescale * (module.max_timescale / module.min_timescale) ** fraction


def rope64(self, inputs, sin, cos):
    first_half, second_half = torch.chunk(inputs, 2, dim=-1)
    return torch.cat((first_half * cos - second_half * sin, second_half * cos + first_half * sin), dim=-1)


dia_layers.RotaryEmbedding.apply_rope = rope64


def without_casts(cls, name, old, new, count):
    # The same method with the float32 casts removed.
    source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    assert source.count(old) == count, (name, source.count(old))
    namespace = {}
    exec(source.replace(old, new), sys.modules[cls.__module__].__dict__, namespace)
    setattr(cls, name, namespace[name])


without_casts(dia_layers.Decoder, "decode_step", "logits_Bx1xCxV.to(torch.float32)", "logits_Bx1xCxV", 1)
without_casts(dia_layers.Decoder, "forward", "logits_BxTxCxV.to(torch.float32)", "logits_BxTxCxV", 1)
without_casts(dia_model.Dia, "_decoder_step", "flat_logits_BCxV.float()", "flat_logits_BCxV", 1)


def hf_norm64(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def hf_rotary64(self, x, position_ids):
    inv_freq = self.inv_freq.double()[None, :, None].expand(position_ids.shape[0], -1, 1)
    freqs = (inv_freq @ position_ids[:, None, :].double()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def hf_attention64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = hf_dia.repeat_kv(key, module.num_key_value_groups)
    value_states = hf_dia.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous(), attn_weights


hf_dia.DiaRMSNorm.forward = hf_norm64
hf_dia.DiaRotaryEmbedding.forward = hf_rotary64
hf_dia.eager_attention_forward = hf_attention64


def closed_form(name, shape):
    n = math.prod(shape)
    i = torch.arange(n, dtype=torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((torch.arange(n) ** 2) % 9973).double())
    if name.endswith("norm.weight"):
        return 1 + 0.2 * s
    if "embedding" in name:
        return 0.5 * s
    fan_in = shape[0] * shape[1] if name.endswith("o_proj.weight") else shape[0]
    return 0.5 * s / math.sqrt(fan_in)


def to_hf(state):
    # The release's DenseGeneral kernels [in..., out...] as transformers' linear weights [out, in].
    hf = {}
    for name, w in state.items():
        if name.startswith("decoder.embeddings."):
            continue
        if name == "decoder.logits_dense.weight":
            hf["logits_dense.weight"] = w.reshape(w.shape[0], -1).T.contiguous()
            continue
        port = "model." + name
        if name.endswith("o_proj.weight"):
            w = w.reshape(w.shape[0] * w.shape[1], -1).T
        elif name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "wi_fused.weight")):
            w = w.reshape(w.shape[0], -1).T
        elif name.endswith("wo.weight"):
            w = w.T
        port = port.replace("mlp.wi_fused", "mlp.gate_up_proj").replace("mlp.wo", "mlp.down_proj")
        hf[port] = w.contiguous()
    hf["model.decoder.embeddings.embed.weight"] = torch.cat([state[f"decoder.embeddings.{c}.weight"] for c in range(CHANNELS)], 0)
    return hf


def main():
    enc = EncoderConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64)
    dec = DecoderConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, cross_hidden_size=16, cross_num_attention_heads=2, cross_num_key_value_heads=2,
                        cross_head_dim=8, vocab_size=VOCAB, num_channels=CHANNELS, max_position_embeddings=64)
    config = DiaConfig(encoder_config=enc, decoder_config=dec, delay_pattern=DELAYS, bos_token_id=BOS, eos_token_id=EOS,
                       pad_token_id=PAD)
    dia = dia_model.Dia(config, device=torch.device("cpu"), load_dac=False)
    dia.compute_dtype = torch.float64
    dia.model = dia.model.double().eval()
    for module in dia.model.modules():
        if isinstance(module, dia_layers.RotaryEmbedding):
            timescale64(module)
        if hasattr(module, "compute_dtype"):
            module.compute_dtype = torch.float64
        if isinstance(module, (dia_layers.MlpBlock,)):
            module.dtype = torch.float64
    shapes = {}
    with torch.no_grad():
        for name, parameter in dia.model.named_parameters():
            shapes[name] = list(parameter.shape)
            parameter.copy_(closed_form(name, list(parameter.shape)).reshape(parameter.shape))

    text_ids = dia._encode_text(TEXT)
    # The encoder's conditional output over the text alone (padding never reaches a real position).
    with torch.no_grad():
        padded = dia._pad_text_input([text_ids])
        state = dia_model.EncoderInferenceState.new(config, padded)
        encoded = dia.model.encoder(torch.cat([torch.zeros_like(padded[:, 0]), padded[:, 0]]), state)[1, : len(text_ids)]

    prompt = torch.tensor([[(5 * f + 3 * c + 1) % 16 for c in range(CHANNELS)] for f in range(4)])
    steps = []
    decode_step = dia.model.decoder.decode_step

    def recording_step(tokens, state, current_idx):
        logits = decode_step(tokens, state, current_idx)
        steps.append({"uncond": logits[0, -1].reshape(-1).tolist(), "cond": logits[1, -1].reshape(-1).tolist()})
        return logits

    dia.model.decoder.decode_step = recording_step
    with torch.no_grad():
        codes = dia.generate(TEXT, max_tokens=16, cfg_scale=3.0, temperature=0.0, top_p=0.95, cfg_filter_top_k=5,
                             audio_prompt=prompt)

    # transformers' loss: the processor's training layout (BOS, the codes, EOS, delayed with PAD) and labels.
    frames = [[(7 * f + 2 * c + 3) % 16 for c in range(CHANNELS)] for f in range(6)]
    hf_config = HfDiaConfig(
        encoder_config=DiaEncoderConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
                                        num_key_value_heads=2, head_dim=8, vocab_size=256, max_position_embeddings=64),
        decoder_config=DiaDecoderConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=8, cross_hidden_size=16, cross_num_attention_heads=2,
                                        cross_num_key_value_heads=2, cross_head_dim=8, vocab_size=VOCAB, num_channels=CHANNELS,
                                        max_position_embeddings=64),
        delay_pattern=DELAYS, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, attn_implementation="eager")
    hf = DiaForConditionalGeneration(hf_config).double().eval()
    for module in hf.modules():
        if isinstance(module, hf_dia.DiaRotaryEmbedding):   # its inverse frequencies are float32; the release's in float64
            half = module.inv_freq.shape[0]
            module.inv_freq = 1.0 / (10000.0 ** (2.0 * torch.arange(half, dtype=torch.float64) / (2 * half)))
    missing, unexpected = hf.load_state_dict(to_hf({n: p.detach() for n, p in dia.model.named_parameters()}), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    audio = torch.tensor(frames)[None]
    audio = torch.nn.functional.pad(audio, (0, 0, 0, 1, 0, 0), value=EOS)
    audio = torch.nn.functional.pad(audio, (0, 0, 1, 0, 0, 0), value=BOS)
    seq_len = audio.shape[1] + max(DELAYS)
    prefill = torch.full((1, seq_len, CHANNELS), PAD, dtype=torch.long)
    prefill[:, : audio.shape[1]] = audio
    index = DiaProcessor.build_indices(bsz=1, seq_len=seq_len, num_channels=CHANNELS, delay_pattern=DELAYS, revert=False)
    decoder_input_ids = DiaProcessor.apply_audio_delay(audio=prefill, pad_token_id=PAD, bos_token_id=BOS, precomputed_idx=index)
    labels = decoder_input_ids.clone()[:, 1:]
    labels[labels == PAD] = -100
    labels[labels == BOS] = -100
    labels = labels.transpose(1, 2).reshape(CHANNELS, -1).contiguous().long()
    with torch.no_grad():
        out = hf(input_ids=text_ids[None], decoder_input_ids=decoder_input_ids[:, :-1], labels=labels)
        hf_encoded = hf.model.encoder(input_ids=text_ids[None]).last_hidden_state[0]

    result = {
        "shapes": shapes, "text": TEXT, "text_ids": text_ids.tolist(), "encoded": encoded.reshape(-1).tolist(),
        "prompt": prompt.tolist(), "steps": steps, "generated": codes.tolist(), "frames": frames,
        "decoder_input_ids": decoder_input_ids[0].tolist(), "labels": labels.tolist(), "loss": float(out.loss),
        "hf_encoder_difference": float((hf_encoded - encoded).abs().max()),
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "dia_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(len(text_ids), len(steps), codes.shape, codes.tolist(), result["loss"], result["hf_encoder_difference"])


if __name__ == "__main__":
    main()
