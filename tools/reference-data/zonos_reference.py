"""Reference fixture for Zonos (Zyphra 2025): Zyphra/Zonos at bc40d98 (zonos.model.Zonos with its prefix conditioner)
at a tiny size, for both backbones: the transformer (zonos/backbone/_torch.py, as the release routes a transformer
checkpoint) and the hybrid (zonos/backbone/_mamba_ssm.py over mamba_ssm 2.2.4's Mamba2 and MHA blocks), in float64.

The tiny transformer has 2 attention layers; the hybrid has 3 layers, the middle one attention: width 32, 4 query heads over
2 key-value heads of 8 (rotary over all 8: adjacent pairs in the transformer, halves in the hybrid, as their configs set),
gated feed-forward 128; Mamba2 with d_state 8, head width 8 (8 heads over the inner width 64), one group, chunk size 4. The
conditioners are the releases' (7 for the transformer, 11 for the hybrid) at width 32. Every parameter and buffer is a closed
form of its name and flat index i: s = sin(0.37 i + 0.11 len(name) + 0.0013 (i^2 mod 9973)); norm weights and D are 1 + 0.2 s,
A_log 0.3 s, other matrices 0.5 s / sqrt(columns), everything else 0.5 s. Codebook 0's end-token row is then scaled so the
greedy generation ends early (the factor is recorded); a second generation, with that row zeroed, runs to the length limit.

mamba_ssm's kernels need CUDA (triton, causal_conv1d, flash_attn); they are replaced by the same arithmetic in PyTorch:
mamba_chunk_scan_combined by the sequential recurrence with its final state, the gated RMSNorm by rms_norm_ref's arithmetic in
float64 (it casts the weight to float32), layer_norm_fn by LayerNorm on the summed residual, and flash_attn's rotary embedding and
flash_attn_with_kvcache by their definitions (the rotary embedding at each sample's cache length, then causal attention over
the cache). The transformer's rotary tables and the torch attention are computed in float64, and the inference caches are
allocated in float64 instead of bfloat16, the guided logits are not cast to float32, and Mamba2's A is not computed in float32. eSpeak is not called: the espeak conditioner's phonemes are given.

Records, for each backbone: the phoneme ids of the given phonemes; the conditioning (conditional, then unconditional:
text only) for a cond_dict with a speaker embedding; a greedy generation (CFG 2, repetition penalty 3 over 2 frames) after a
3-frame audio prefix: every guided logit vector and the returned codes; and a training loss over a 5-frame recording (the
codes and an end frame, delayed; each head's next delayed token, mask tokens unscored), computed with the release's modules.

Run with ~/source/repos/.venv-zonos (torch 2.6.0 cpu, einops) with refsrc/Zonos and refsrc/mamba (v2.2.4).
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/zonos_reference.json.
"""
import importlib.machinery
import json
import math
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.join(HERE, "..", "..", "..", "refsrc")


# ---- CUDA-only packages: importable stand-ins ----------------------------------------------------------------------
class _Anything:
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Anything()

    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else _Anything()


class _Config:
    def __init__(self, kwargs=None, num_warps=4, num_stages=2, **rest):
        self.kwargs, self.num_warps, self.num_stages = kwargs or {}, num_warps, num_stages


def _attr(k):
    if k.startswith("__"):
        raise AttributeError(k)
    return _Anything()


triton = types.ModuleType("triton")
triton.__version__ = "3.1.0"
language = types.ModuleType("triton.language")
for module in (triton, language):
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
    module.__getattr__ = _attr
triton.Config, triton.language = _Config, language
sys.modules.update({"triton": triton, "triton.language": language, "selective_scan_cuda": types.ModuleType("selective_scan_cuda")})


def _rotate(x, cos, sin, interleaved):
    # x [..., d]; cos and sin [..., d/2] broadcastable: flash_attn's apply_rotary_emb on the first 2 * cos.shape[-1] dims.
    rd = cos.shape[-1] * 2
    xr, rest = x[..., :rd], x[..., rd:]
    if interleaved:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)
    else:
        x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([out, rest], dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """flash_attn.layers.rotary.RotaryEmbedding (no xPos scaling) by its definition."""

    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None, pos_idx_in_fp32=True, device=None):
        super().__init__()
        self.dim, self.base, self.interleaved = dim, float(base), interleaved
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        freqs = torch.outer(torch.arange(seqlen, dtype=torch.float64), self.inv_freq)
        self._cos_cached, self._sin_cached = torch.cos(freqs), torch.sin(freqs)

    def forward(self, q, kv, seqlen_offset=0, max_seqlen=None):
        seqlen = q.shape[1]
        self._update_cos_sin_cache(int(max_seqlen or 0) + seqlen + 64)
        offset = seqlen_offset if torch.is_tensor(seqlen_offset) else torch.tensor([seqlen_offset])
        positions = offset.view(-1, 1) + torch.arange(seqlen).view(1, -1)                     # [b, s]
        cos = self._cos_cached[positions].unsqueeze(2)
        sin = self._sin_cached[positions].unsqueeze(2)
        q = _rotate(q, cos, sin, self.interleaved)
        k = _rotate(kv[:, :, 0], cos, sin, self.interleaved)
        return q, torch.stack([k, kv[:, :, 1]], dim=2)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None,
                            softmax_scale=None, causal=False, rotary_interleaved=True, **rest):
    """flash_attn_with_kvcache by its definition: the new keys (rotated at each sample's cache length) and values written
    into the caches, then attention over each sample's filled cache, the query at the cache's end."""
    b, s = q.shape[:2]
    lengths = cache_seqlens if torch.is_tensor(cache_seqlens) else torch.full((b,), int(cache_seqlens))
    outputs = []
    for i in range(b):
        start = int(lengths[i])
        positions = torch.arange(start, start + s)
        qi, ki = q[i:i + 1], k[i:i + 1]
        if rotary_cos is not None:
            cos, sin = rotary_cos[positions].unsqueeze(1), rotary_sin[positions].unsqueeze(1)
            qi = _rotate(qi, cos, sin, rotary_interleaved)
            ki = _rotate(ki, cos, sin, rotary_interleaved)
        k_cache[i, start:start + s] = ki[0]
        v_cache[i, start:start + s] = v[i]
        keys, values = k_cache[i:i + 1, : start + s], v_cache[i:i + 1, : start + s]
        groups = q.shape[2] // keys.shape[2]
        keys, values = keys.repeat_interleave(groups, dim=2), values.repeat_interleave(groups, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", qi, keys) * (softmax_scale or 1 / math.sqrt(q.shape[-1]))
        mask = torch.arange(start + s).view(1, -1) <= (start + torch.arange(s)).view(-1, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
        outputs.append(torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), values))
    return torch.cat(outputs)


flash_attn = types.ModuleType("flash_attn")
flash_attn.flash_attn_with_kvcache = flash_attn_with_kvcache
flash_layers = types.ModuleType("flash_attn.layers")
flash_rotary = types.ModuleType("flash_attn.layers.rotary")
flash_rotary.RotaryEmbedding = RotaryEmbedding
sys.modules.update({"flash_attn": flash_attn, "flash_attn.layers": flash_layers, "flash_attn.layers.rotary": flash_rotary})

# Japanese normalization's packages are imported at module level; it is not exercised here.
kanjize = types.ModuleType("kanjize")
kanjize.number2kanji = lambda n: str(n)
sudachipy = types.ModuleType("sudachipy")
sudachipy.Dictionary = lambda **k: types.SimpleNamespace(create=lambda: None)
sudachipy.SplitMode = types.SimpleNamespace(A="A")
for module in (kanjize, sudachipy, flash_attn, flash_layers, flash_rotary):
    module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
sys.modules.update({"kanjize": kanjize, "sudachipy": sudachipy})

sys.path.insert(0, os.path.join(REFSRC, "mamba"))
sys.path.insert(0, os.path.join(REFSRC, "Zonos"))
torch.set_default_dtype(torch.float64)


class DACStub:
    """The codec is not run: generation returns codes."""
    codebook_size, num_codebooks, sampling_rate = 1024, 9, 44100


import zonos.autoencoder  # noqa: E402
zonos.autoencoder.DACAutoencoder = DACStub
import mamba_ssm.modules.block as mamba_block  # noqa: E402
import mamba_ssm.modules.mamba2 as mamba2  # noqa: E402
import mamba_ssm.modules.mha as mamba_mha  # noqa: E402
import mamba_ssm.ops.triton.layernorm_gated as layernorm_gated  # noqa: E402
import zonos.backbone._mamba_ssm as zonos_mamba  # noqa: E402
import zonos.backbone._torch as zonos_torch  # noqa: E402
import zonos.conditioning as conditioning  # noqa: E402
import zonos.model as zonos_model  # noqa: E402
from zonos.config import BackboneConfig, PrefixConditionerConfig, ZonosConfig  # noqa: E402

mamba_mha.flash_attn_with_kvcache = flash_attn_with_kvcache
mamba_mha.RotaryEmbedding = RotaryEmbedding
mamba2.selective_state_update = None
mamba2.causal_conv1d_fn = mamba2.causal_conv1d_update = None


def layer_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6, is_rms_norm=False, **rest):
    summed = x + residual if residual is not None else x
    out = F.layer_norm(summed, (summed.shape[-1],), weight, bias, eps)
    return (out, summed) if prenorm else out


mamba_block.layer_norm_fn = layer_norm_fn
zonos_mamba.layer_norm_fn = layer_norm_fn


def chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None,
               dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False):
    """mamba_chunk_scan_combined's recurrence, step by step: x [b, l, h, p], dt [b, l, h], B and C [b, l, g, n]."""
    b, l, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    heads_per_group = h // g
    state = torch.zeros(b, h, p, n) if initial_states is None else initial_states.clone()
    ys = []
    for t in range(l):
        Bt = B[:, t].repeat_interleave(heads_per_group, dim=1)                                  # [b, h, n]
        Ct = C[:, t].repeat_interleave(heads_per_group, dim=1)
        dA = torch.exp(dt[:, t] * A)                                                             # [b, h]
        state = state * dA[:, :, None, None] + torch.einsum("bh,bhp,bhn->bhpn", dt[:, t], x[:, t], Bt)
        y = torch.einsum("bhpn,bhn->bhp", state, Ct)
        if D is not None:
            y = y + x[:, t] * (D[:, None] if D.dim() == 1 else D)
        ys.append(y)
    y = torch.stack(ys, dim=1)
    return (y, state) if return_final_states else y


mamba2.mamba_chunk_scan_combined = chunk_scan
def gated_rms_norm(self, x, z=None):
    # rms_norm_ref's arithmetic (which casts the weight to float32) in float64: x · silu(z), then RMS over each group.
    if z is not None and not self.norm_before_gate:
        x = x * F.silu(z)
    group = x.reshape(*x.shape[:-1], -1, self.group_size)
    out = (group * torch.rsqrt(group.square().mean(dim=-1, keepdim=True) + self.eps)).reshape(x.shape) * self.weight
    if self.bias is not None:
        out = out + self.bias
    if z is not None and self.norm_before_gate:
        out = out * F.silu(z)
    return out


layernorm_gated.RMSNorm.forward = gated_rms_norm


def freqs64(seq_len, n_elem, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float64)[: (n_elem // 2)] / n_elem))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float64), freqs)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def rotary64(x, freqs_cis):
    xshaped = x.reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    out = torch.stack([xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                       xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1]], -1)
    return out.flatten(3)


zonos_torch.precompute_freqs_cis = freqs64
zonos_torch.apply_rotary_emb = rotary64
SDPA = F.scaled_dot_product_attention


def attention64(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) * (scale if scale is not None else 1 / math.sqrt(q.shape[-1]))
    if is_causal:
        # Bottom-right aligned, so a query block at the end of the keys sees every earlier key (the cached prefill and
        # training calls have as many queries as keys, where the alignments agree).
        lq, lk = q.shape[-2], k.shape[-2]
        mask = torch.arange(lk).view(1, -1) <= (lk - lq + torch.arange(lq)).view(-1, 1)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


F.scaled_dot_product_attention = attention64
def _without_float32_logits():
    # _compute_logits ends its head with .float(); the same method without the cast.
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(zonos_model.Zonos._compute_logits))
    assert source.count(".squeeze(2).float()") == 1
    namespace = {}
    exec(source.replace(".squeeze(2).float()", ".squeeze(2)"), zonos_model.__dict__, namespace)
    zonos_model.Zonos._compute_logits = namespace["_compute_logits"]


_without_float32_logits()


def _without_float32_a():
    # Mamba2.forward and Mamba2.step take A = -exp(A_log.float()); the same methods without the cast.
    import inspect
    import textwrap
    for name in ("forward", "step"):
        source = textwrap.dedent(inspect.getsource(getattr(mamba2.Mamba2, name)))
        assert source.count("self.A_log.float()") >= 1, name
        namespace = {}
        exec(source.replace("self.A_log.float()", "self.A_log"), mamba2.__dict__, namespace)
        setattr(mamba2.Mamba2, name, namespace[name])


_without_float32_a()
_setup_cache = zonos_model.Zonos.setup_cache
zonos_model.Zonos.setup_cache = lambda self, batch_size, max_seqlen, dtype=None: _setup_cache(self, batch_size, max_seqlen, torch.float64)

PHONEMES = "ðɪs ɪz zˈoʊnəs, spˈiːkɪŋ!"


def closed_form(name, shape):
    n = math.prod(shape)
    i = torch.arange(n, dtype=torch.float64)
    s = torch.sin(0.37 * i + 0.11 * len(name) + 0.0013 * ((torch.arange(n) ** 2) % 9973).double())
    if name.endswith("norm.weight") or name.endswith("norm_f.weight") or name.endswith("norm2.weight") or name.endswith(".D"):
        return 1 + 0.2 * s
    if name.endswith("A_log"):
        return 0.3 * s
    if len(shape) == 2:
        return 0.5 * s / math.sqrt(shape[1])
    return 0.5 * s


TRANSFORMER_CONDITIONERS = [
    {"type": "EspeakPhonemeConditioner", "name": "espeak"},
    {"cond_dim": 128, "uncond_type": "learned", "projection": "linear", "type": "PassthroughConditioner", "name": "speaker"},
    {"input_dim": 8, "uncond_type": "learned", "type": "FourierConditioner", "name": "emotion"},
    {"min_val": 0, "max_val": 24000, "uncond_type": "learned", "type": "FourierConditioner", "name": "fmax"},
    {"min_val": 0, "max_val": 400, "uncond_type": "learned", "type": "FourierConditioner", "name": "pitch_std"},
    {"min_val": 0, "max_val": 40, "uncond_type": "learned", "type": "FourierConditioner", "name": "speaking_rate"},
    {"min_val": -1, "max_val": 126, "uncond_type": "learned", "type": "IntegerConditioner", "name": "language_id"},
]
HYBRID_CONDITIONERS = TRANSFORMER_CONDITIONERS + [
    {"input_dim": 8, "min_val": 0.5, "max_val": 0.8, "uncond_type": "learned", "type": "FourierConditioner", "name": "vqscore_8"},
    {"min_val": -1.0, "max_val": 1000, "uncond_type": "learned", "type": "FourierConditioner", "name": "ctc_loss"},
    {"min_val": 1, "max_val": 5, "uncond_type": "learned", "type": "FourierConditioner", "name": "dnsmos_ovrl"},
    {"min_val": 0, "max_val": 1, "uncond_type": "learned", "type": "IntegerConditioner", "name": "speaker_noised"},
]


def build(hybrid):
    attn = {"causal": True, "num_heads": 4, "num_heads_kv": 2, "rotary_emb_dim": 8, "qkv_proj_bias": False, "out_proj_bias": False}
    if hybrid:
        backbone = BackboneConfig(d_model=32, d_intermediate=0, attn_mlp_d_intermediate=128, n_layer=3,
                                  ssm_cfg={"layer": "Mamba2", "d_state": 8, "headdim": 8, "chunk_size": 4},
                                  attn_layer_idx=[1], attn_cfg=attn)
        cls = zonos_mamba.MambaSSMZonosBackbone
    else:
        backbone = BackboneConfig(d_model=32, d_intermediate=0, attn_mlp_d_intermediate=128, n_layer=2, ssm_cfg={},
                                  attn_layer_idx=[0, 1], attn_cfg={**attn, "rotary_emb_interleaved": True})
        cls = zonos_torch.TorchZonosBackbone
    config = ZonosConfig(backbone, PrefixConditionerConfig(HYBRID_CONDITIONERS if hybrid else TRANSFORMER_CONDITIONERS, "linear"))
    model = zonos_model.Zonos(config, cls).double().eval()
    state, shapes = model.state_dict(), {}
    with torch.no_grad():
        for name, tensor in state.items():
            if not tensor.is_floating_point():
                continue
            shapes[name] = list(tensor.shape)
            tensor.copy_(closed_form(name, list(tensor.shape)).reshape(tensor.shape))
    return model, shapes


def run(hybrid):
    model, shapes = build(hybrid)
    conditioning.phonemize = lambda texts, languages: [PHONEMES for _ in texts]
    phoneme_ids, _ = conditioning.tokenize_phonemes([PHONEMES])
    speaker = closed_form("speaker_input", [128]).view(1, 1, 128)
    cond_dict = conditioning.make_cond_dict(text="unused", language="en-us", speaker=speaker, device="cpu")
    with torch.no_grad():
        prefix = model.prepare_conditioning(cond_dict)
    prefix_codes = torch.tensor([[[(37 * f + 101 * k + 5) % 1024 for f in range(3)] for k in range(9)]])

    def generate(record):
        logits_seen = []
        compute = model._compute_logits

        def recording(hidden_states, inference_params, cfg_scale):
            logits = compute(hidden_states, inference_params, cfg_scale)
            logits_seen.append(logits[0].reshape(-1).tolist())
            return logits

        model._compute_logits = recording
        with torch.no_grad():
            codes = model.generate(prefix, prefix_codes, max_new_tokens=10, cfg_scale=2.0, sampling_params={"temperature": 0.0},
                                   progress_bar=False, disable_torch_compile=True)
        model._compute_logits = compute
        return codes, logits_seen

    # Scale codebook 0's end row until the greedy generation stops early.
    end_row = model.heads[0].weight[1024].clone()
    factor = 1.0
    for factor in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]:
        with torch.no_grad():
            model.heads[0].weight[1024] = end_row * factor
        codes, logits_seen = generate(False)
        if codes.shape[2] < 3 + 10 and codes.shape[2] > 3 + 2:
            break

    # A generation that never ends (codebook 0's end row zeroed) runs to the length limit, through the delay pattern's end.
    with torch.no_grad():
        model.heads[0].weight[1024] = 0
    full_codes, full_logits = generate(False)
    with torch.no_grad():
        model.heads[0].weight[1024] = end_row * factor

    # The training loss: the cond branch's prefix, then the delayed codes and end frame, each head scored on the next
    # delayed token (mask tokens unscored).
    frames = torch.tensor([[[(11 * f + 7 * k + 3) % 1024 for f in range(5)] for k in range(9)]])
    with_end = torch.cat([frames, torch.full((1, 9, 1), 1024)], dim=2)
    delayed = zonos_model.apply_delay_pattern(with_end, model.masked_token_id)                  # [1, 9, T + 1 + 9]
    with torch.no_grad():
        inputs = torch.cat([prefix[:1], model.embed_codes(delayed[..., :-1])], dim=1)
        params = model.setup_cache(batch_size=1, max_seqlen=inputs.shape[1] + 8)
        hidden = model.backbone(inputs, params)[:, prefix.shape[1]:]
        logits = model.apply_heads(hidden)                                                        # [1, 9, L, 1025]
        targets = delayed[..., 1:]
        mask = targets != model.masked_token_id
        loss = F.cross_entropy(logits[mask], targets[mask])

    return {
        "shapes": shapes, "phonemes": PHONEMES, "phoneme_ids": phoneme_ids[0].tolist(),
        "language_id": int(cond_dict["language_id"].item()), "speaker": speaker.reshape(-1).tolist(),
        "prefix": prefix.reshape(-1).tolist(), "prefix_length": prefix.shape[1], "prefix_codes": prefix_codes[0].tolist(),
        "end_factor": factor, "generated": codes[0].tolist(), "logits": logits_seen,
        "full_generated": full_codes[0].tolist(), "full_logits": full_logits,
        "frames": frames[0].tolist(), "loss": float(loss),
    }


def main():
    result = {"transformer": run(False), "hybrid": run(True)}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "zonos_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    for k, v in result.items():
        print(k, v["prefix_length"], v["end_factor"], len(v["generated"][0]), len(v["logits"]), v["loss"])


if __name__ == "__main__":
    main()
