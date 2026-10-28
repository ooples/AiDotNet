"""Reference fixture for CSM (Sesame 2025): SesameAILabs/csm at daed31e (models.Model with torchtune 0.4.0's llama3_2
builder, as the release defines it) at a tiny size, and Hugging Face's CsmForConditionalGeneration (transformers 4.52.4,
the port written with Sesame and the only training code) on the same weights, in float64.

The two flavors are llama3_2(num_layers=1, num_heads=2, num_kv_heads=1, embed_dim=16, intermediate_dim=32, rope_base=500000,
scale_factor=32) for the backbone and the decoder alike; the text vocabulary is Llama 3's 128,256, the audio vocabulary 19
over 4 codebooks. Every parameter is a closed form of its release name and flat index i: s = sin(0.37 i + 0.11 len(name) +
0.0013 (i^2 mod 9973)); RMSNorm scales are 1 + 0.2 s, everything else 0.5 s / sqrt(fan-in) for matrices and 0.5 s for
the embedding tables and audio_head (whose rows are not fan-in scaled).

torchtune computes its rotary embedding and RMSNorm in float32 and casts the decoders' output to float32, Hugging Face its rotary embedding, RMSNorm and eager softmax in float32
(its inverse frequencies are set from torchtune's float64 ones) and torch 2.4's CPU scaled_dot_product_attention returns NaN for float64 with a boolean mask; each is replaced by the same
arithmetic in float64.

Records: the backbone's final hidden states and codebook-0 logits over a conversation's frames (two turns' text, the first
turn's audio frames and its frame of zeros); the frames Model.generate_frame samples with top-k 1 (greedy) after it, with every logit vector it samples from;
and Hugging Face's loss, backbone loss and decoder loss for those frames with codebook-0 labels on the audio frames and the
decoder trained on two of them. The release's weights are converted to Hugging Face's names (torchtune's interleaved
rotary pairs reordered to halves) and loaded strictly.

Run with ~/source/repos/.venv-csm (torch 2.4.0, torchtune 0.4.0, transformers 4.52.4) and the csm clone (refsrc/csm).
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/csm_reference.json.
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "csm"))
torch.set_default_dtype(torch.float64)

import torch.nn.functional as F  # noqa: E402
import torchtune.modules.rms_norm as tt_norm  # noqa: E402
import torchtune.models.llama3_1._position_embeddings as tt_rope  # noqa: E402
from torchtune.models import llama3_2  # noqa: E402
import transformers.models.csm.modeling_csm as hf_csm  # noqa: E402
from transformers import CsmConfig, CsmForConditionalGeneration  # noqa: E402

import models  # noqa: E402  (refsrc/csm/models.py)

TEXT_VOCAB, AUDIO_VOCAB, CODEBOOKS, WIDTH = 128256, 19, 4, 16
SDPA = F.scaled_dot_product_attention


def attention64(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if q.dtype != torch.float64:
        return SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
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


def rope_init64(self):
    freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].double() / self.dim))
    theta = self.apply_scaling(freqs, self.scale_factor, self.low_freq_factor, self.high_freq_factor, self.old_context_len)
    self.register_buffer("theta", theta, persistent=False)
    seq_idx = torch.arange(self.max_seq_len, dtype=torch.float64)
    idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta)
    self.register_buffer("cache", torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1), persistent=False)
    self.is_cache_built = True


def rope_forward64(self, x, *, input_pos=None):
    seq_len = x.size(1)
    rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out = torch.stack([xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                         xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1]], -1)
    return x_out.flatten(3)


def tt_norm64(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


def hf_norm64(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def hf_rotary64(self, x, position_ids):
    inv_freq = self.inv_freq.double()[None, :, None].expand(position_ids.shape[0], -1, 1)
    freqs = (inv_freq @ position_ids[:, None, :].double()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def without_float32_output():
    # TransformerDecoder.forward ends with self.output(h).float(); the same method without the cast.
    import inspect
    import textwrap
    import torchtune.modules.transformer as tt_transformer
    source = textwrap.dedent(inspect.getsource(tt_transformer.TransformerDecoder.forward))
    assert source.count("self.output(h).float()") == 1
    namespace = {}
    exec(source.replace("self.output(h).float()", "self.output(h)"), tt_transformer.__dict__, namespace)
    tt_transformer.TransformerDecoder.forward = namespace["forward"]


without_float32_output()
def hf_attention64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = hf_csm.repeat_kv(key, module.num_key_value_groups)
    value_states = hf_csm.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous(), attn_weights


hf_csm.eager_attention_forward = hf_attention64
tt_rope.Llama3ScaledRoPE.rope_init = rope_init64
tt_rope.Llama3ScaledRoPE.forward = rope_forward64
tt_norm.RMSNorm.forward = tt_norm64
hf_csm.CsmRMSNorm.forward = hf_norm64
hf_csm.CsmRotaryEmbedding.forward = hf_rotary64


def tiny():
    return llama3_2.llama3_2(vocab_size=8, num_layers=1, num_heads=2, num_kv_heads=1, embed_dim=WIDTH, max_seq_len=64,
                             intermediate_dim=32, attn_dropout=0.0, norm_eps=1e-5, rope_base=500_000, scale_factor=32)


models.FLAVORS["tiny"] = tiny


def closed_form(name, shape):
    count = math.prod(shape)
    k = torch.arange(count, dtype=torch.int64)
    s = torch.sin(0.37 * k.to(torch.float64) + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith(".scale"):
        return 1 + 0.2 * s
    if name.endswith(("embeddings.weight", "audio_head")):
        return 0.5 * s
    return 0.5 * s / math.sqrt(count / shape[0])


def halves(weight, head_dim):
    out, inputs = weight.shape
    w = weight.view(out // head_dim, head_dim // 2, 2, inputs)
    return w.transpose(1, 2).reshape(out, inputs)


def to_hf(state):
    hd = WIDTH // 2
    hf = {}
    for name, value in state.items():
        for release, port in (("backbone.", "backbone_model."), ("decoder.", "depth_decoder.model.")):
            if name.startswith(release):
                rest = name[len(release):]
                rest = (rest.replace("sa_norm.scale", "input_layernorm.weight").replace("mlp_norm.scale", "post_attention_layernorm.weight")
                        .replace("attn.output_proj", "attn.o_proj").replace("attn.", "self_attn.")
                        .replace("mlp.w1", "mlp.gate_proj").replace("mlp.w3", "mlp.up_proj").replace("mlp.w2", "mlp.down_proj")
                        .replace("norm.scale", "norm.weight"))
                if rest.endswith(("q_proj.weight", "k_proj.weight")):
                    value = halves(value, hd)
                hf[port + rest] = value
                break
        else:
            port = {"text_embeddings.weight": "embed_text_tokens.weight",
                    "audio_embeddings.weight": "backbone_model.embed_tokens.embed_audio_tokens.weight",
                    "projection.weight": "depth_decoder.model.inputs_embeds_projector.weight",
                    "codebook0_head.weight": "lm_head.weight", "audio_head": "depth_decoder.codebooks_head.weight"}[name]
            hf[port] = value
    hf["depth_decoder.model.embed_tokens.weight"] = hf["backbone_model.embed_tokens.embed_audio_tokens.weight"]
    return hf


def main():
    model = models.Model(models.ModelArgs(backbone_flavor="tiny", decoder_flavor="tiny", text_vocab_size=TEXT_VOCAB,
                                          audio_vocab_size=AUDIO_VOCAB, audio_num_codebooks=CODEBOOKS)).eval()
    shapes = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            shapes[name] = list(parameter.shape)
            parameter.copy_(closed_form(name, list(parameter.shape)).reshape(parameter.shape))

    first_text = [128000, 58, 15, 60, 9906, 1917, 128001]
    second_text = [128000, 58, 16, 60, 13347, 128001]
    audio = [[(3 * f + 5 * k + 1) % 16 for k in range(CODEBOOKS)] for f in range(5)]
    rows, masks = [], []
    for t in first_text:
        rows.append([0] * CODEBOOKS + [t])
        masks.append([False] * CODEBOOKS + [True])
    for frame in audio + [[0] * CODEBOOKS]:
        rows.append(frame + [0])
        masks.append([True] * CODEBOOKS + [False])
    conversation = len(rows)
    for t in second_text:
        rows.append([0] * CODEBOOKS + [t])
        masks.append([False] * CODEBOOKS + [True])
    tokens = torch.tensor(rows)[None]
    mask = torch.tensor(masks)[None]

    with torch.no_grad():
        embeds = (model._embed_tokens(tokens) * mask.unsqueeze(-1)).sum(dim=2)
        hidden = model.backbone(embeds)
        logits0 = model.codebook0_head(hidden)

        sampled_logits = []
        sample_topk = models.sample_topk

        def recording_sample(logits, topk, temperature):
            sampled_logits.append(logits[0].tolist())
            return sample_topk(logits, topk, temperature)

        models.sample_topk = recording_sample
        model.setup_caches(1)
        curr_tokens, curr_mask = tokens, mask
        curr_pos = torch.arange(tokens.size(1))[None]
        generated = []
        for _ in range(6):
            sample = model.generate_frame(curr_tokens, curr_mask, curr_pos, 0.9, 1)
            if torch.all(sample == 0):
                break
            generated.append(sample[0].tolist())
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long()], dim=1).unsqueeze(1)
            curr_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool()], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # The training loss over the first turn: its text, its audio frames and the frame of zeros.
        train_rows = len(first_text) + len(audio) + 1
        decoder_frames = [len(first_text) + 1, len(first_text) + 3]
        labels = torch.full((1, train_rows, CODEBOOKS), -100)
        for f, frame in enumerate(audio):
            p = len(first_text) + f
            labels[0, p] = torch.tensor(frame)
            if p not in decoder_frames:
                labels[0, p, 1:] = -100
        backbone = dict(num_codebooks=CODEBOOKS, vocab_size=AUDIO_VOCAB, text_vocab_size=TEXT_VOCAB, hidden_size=WIDTH,
                        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        rope_theta=500000, rms_norm_eps=1e-5, max_position_embeddings=64,
                        rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                      "original_max_position_embeddings": 8192})
        depth = dict(num_codebooks=CODEBOOKS, backbone_hidden_size=WIDTH, vocab_size=AUDIO_VOCAB, hidden_size=WIDTH,
                     intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                     rope_theta=500000, rms_norm_eps=1e-5, max_position_embeddings=CODEBOOKS + 1,
                     rope_scaling=backbone["rope_scaling"])
        # The codec (Hugging Face's default Mimi) is built but never run: the inputs are embeddings.
        config = CsmConfig(**backbone, depth_decoder_config=depth, attn_implementation="eager")
        hf = CsmForConditionalGeneration(config).double().eval()
        hf_state = to_hf({name: p.detach() for name, p in model.named_parameters()})
        theta = model.backbone.layers[0].attn.pos_embeddings.theta   # the same Llama 3 scaled frequencies, in float64
        for module in hf.modules():
            if isinstance(module, hf_csm.CsmRotaryEmbedding):
                module.inv_freq = theta.clone()
        missing, unexpected = hf.load_state_dict(hf_state, strict=False)
        missing = [m for m in missing if not m.startswith("codec_model.")]
        assert not missing and not unexpected, (missing, unexpected)
        out = hf(inputs_embeds=embeds[:, :train_rows], labels=labels)
        hf_hidden = hf.backbone_model(inputs_embeds=embeds).last_hidden_state

    result = {
        "text_vocab": TEXT_VOCAB, "audio_vocab": AUDIO_VOCAB, "codebooks": CODEBOOKS, "width": WIDTH,
        "shapes": shapes, "first_text": first_text, "second_text": second_text, "audio": audio,
        "conversation_rows": conversation, "hidden": hidden[0].reshape(-1).tolist(), "logits0": logits0[0].reshape(-1).tolist(),
        "generated": generated, "sampled_logits": sampled_logits, "decoder_frames": decoder_frames, "train_rows": train_rows,
        "loss": float(out.loss), "backbone_loss": float(out.backbone_loss), "decoder_loss": float(out.depth_decoder_loss),
        "hf_hidden_difference": float((hf_hidden - hidden).abs().max()),
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "csm_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(generated, len(sampled_logits), result["loss"], result["backbone_loss"], result["decoder_loss"], result["hf_hidden_difference"])


if __name__ == "__main__":
    main()
