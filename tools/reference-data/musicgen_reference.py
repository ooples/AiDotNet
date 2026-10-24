"""MusicGen reference: audiocraft's LMModel (models/lm.py) with the delay pattern and text cross-attention.

A tiny configuration of the reference language model (4 codebooks of 16 codes, D = 16, 2 heads, 2 layers, the
MusicGen settings: pre-norm, bias-free linears, cross-attention, and the paper's ReLU feed-forward) is conditioned on
a tiny Hugging Face T5 encoder and audiocraft's text projection (T5Conditioner.output_proj). The fixture records:
- the weights (LM state dict, T5 encoder state dict, output_proj), float32;
- the conditioning of a token sequence, the logits and validity mask of compute_predictions for a code sequence,
  and the training cross-entropy (MusicGenSolver._compute_cross_entropy);
- the conditioning again with T5LayerNorm's mean square taken in float64: Hugging Face computes it in float32 even
  in a float64 model, which puts ~1e-7 of rounding into the stock conditioning;
- codes generated greedily with classifier-free guidance 3.0, following LMModel.generate's loop;
- the melody model (§2.3): audiocraft's ChromaExtractor (12 bins, 2^14 window, argmax) on a deterministic tone that
  moves from C4 to A4, its ChromaStemConditioner output projection, and an LMModel whose conditions are a prefix
  (the chroma2music fuser: prepend self_wav, then description; no cross-attention), with the same logits, loss and
  guided greedy generation checks;
- the stereo model (§4.3): eight interleaved codebooks [L0, R0, L1, R1, ...] with the stereo partial delay
  [0, 0, 1, 1, 2, 2, 3, 3], text cross-attention, the same checks.
The reference runs in float64 from the float32 weights.
Pass an audiocraft checkout with --audiocraft. Default: verify; --write: regenerate
tests/AiDotNet.Tests/Audio/ReferenceData/musicgen_reference.json.
"""
import argparse
import json
import os
import sys
import types

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/musicgen_reference.json"
K, CARD, DIM, HEADS, LAYERS, DELAYS = 4, 16, 16, 2, 2, [0, 1, 2, 3]
T5 = dict(vocab_size=64, d_model=8, d_kv=4, d_ff=16, num_layers=2, num_heads=2, relative_attention_num_buckets=32,
          relative_attention_max_distance=128, dropout_rate=0.0, feed_forward_proj="relu", layer_norm_epsilon=1e-6)
TOKENS = [5, 17, 33, 2, 9, 1]
FRAMES, GENERATED = 7, 6


def compute(audiocraft):
    stub = types.ModuleType("xformers")
    stub.ops = types.ModuleType("xformers.ops")
    sys.modules.setdefault("xformers", stub)
    sys.modules.setdefault("xformers.ops", stub.ops)
    sys.path.insert(0, audiocraft)
    import torch
    # The custom (non-memory-efficient) attention path only uses xformers' unbind, which is torch.unbind.
    stub.ops.unbind = torch.unbind
    from transformers import T5Config, T5EncoderModel
    from audiocraft.models.lm import LMModel
    from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
    from audiocraft.modules.conditioners import ConditioningProvider, ConditionFuser

    torch.manual_seed(1234)
    fuser = ConditionFuser(fuse2cond={"cross": ["description"], "prepend": [], "sum": [], "input_interpolate": []})
    lm = LMModel(DelayedPatternProvider(n_q=K, delays=DELAYS), ConditioningProvider({}), fuser, n_q=K, card=CARD, dim=DIM,
                 num_heads=HEADS, num_layers=LAYERS, hidden_scale=4, norm="layer_norm", norm_first=True, bias_proj=False,
                 bias_ff=False, bias_attn=False, cross_attention=True, causal=True, custom=True, activation="relu", dropout=0.0,
                 weight_init="gaussian", depthwise_init="current", zero_bias_init=True, cfg_coef=3.0)
    t5 = T5EncoderModel(T5Config(**T5))
    proj = torch.nn.Linear(T5["d_model"], DIM)
    # Round every weight to float32 (what the fixture stores), then compute in float64.
    for module in (lm, t5, proj):
        with torch.no_grad():
            for p in module.parameters():
                p.copy_(p.float())
        module.double().eval()

    ids = torch.tensor([TOKENS])
    with torch.no_grad():
        hidden = t5(input_ids=ids).last_hidden_state
        embeds = proj(hidden)                                                     # [1, Lc, D]
        from transformers.models.t5 import modeling_t5
        stock_norm = modeling_t5.T5LayerNorm.forward

        def norm64(self, h):
            variance = h.pow(2).mean(-1, keepdim=True)
            return self.weight * (h * torch.rsqrt(variance + self.variance_epsilon))

        modeling_t5.T5LayerNorm.forward = norm64
        exact = proj(t5(input_ids=ids).last_hidden_state)
        modeling_t5.T5LayerNorm.forward = stock_norm
        mask = torch.ones(1, len(TOKENS), dtype=torch.long)
        g = torch.Generator().manual_seed(7)
        codes = torch.randint(0, CARD, (1, K, FRAMES), generator=g)
        out = lm.compute_predictions(codes, [], condition_tensors={"description": (embeds, mask)})
        logits, valid = out.logits, out.mask                                     # [1, K, T, card], [1, K, T]
        ce = torch.zeros([], dtype=torch.float64)
        for k in range(K):
            m = valid[0, k]
            ce += torch.nn.functional.cross_entropy(logits[0, k][m], codes[0, k][m])
        ce = ce / K

        # LMModel.generate's loop with precomputed conditions (cond and null rows), greedy, CFG 3.0.
        cfg = {"description": (torch.cat([embeds, torch.zeros_like(embeds)]), torch.cat([mask, torch.zeros_like(mask)]))}
        pattern = lm.pattern_provider.get_pattern(GENERATED)
        gen = torch.full((1, K, GENERATED), -1, dtype=torch.long)
        seq, _, smask = pattern.build_pattern_sequence(gen, lm.special_token_id)
        start = pattern.get_first_step_with_timesteps(0)
        with lm.streaming():
            uncond_state = lm.get_streaming_state()
            prev = 0
            for offset in range(start, seq.shape[-1]):
                nxt = lm._sample_next_token(seq[..., prev:offset], cfg, uncond_state, use_sampling=False, cfg_coef=3.0)
                valid_step = smask[..., offset:offset + 1].expand(1, -1, -1)
                nxt[~valid_step] = lm.special_token_id
                seq[..., offset:offset + 1] = torch.where(seq[..., offset:offset + 1] == -1, nxt, seq[..., offset:offset + 1])
                prev = offset
        generated, _, _ = pattern.revert_pattern_sequence(seq, special_token=-1)

    # ---- the melody model ----
    from audiocraft.modules.chroma import ChromaExtractor
    import math
    sr, n = 32000, 24000
    tone = torch.tensor([[0.5 * math.sin(2 * math.pi * (261.63 if i < n // 2 else 440.0) * i / sr) for i in range(n)]],
                        dtype=torch.float64)
    extractor = ChromaExtractor(sample_rate=sr, n_chroma=12, radix2_exp=14, argmax=True).double()
    chroma = extractor(tone)                                                       # [1, frames, 12]
    torch.manual_seed(99)
    melody_fuser = ConditionFuser(fuse2cond={"cross": [], "prepend": ["self_wav", "description"], "sum": [],
                                             "input_interpolate": []})
    lm_melody = LMModel(DelayedPatternProvider(n_q=K, delays=DELAYS), ConditioningProvider({}), melody_fuser, n_q=K, card=CARD,
                        dim=DIM, num_heads=HEADS, num_layers=LAYERS, hidden_scale=4, norm="layer_norm", norm_first=True,
                        bias_proj=False, bias_ff=False, bias_attn=False, cross_attention=False, causal=True, custom=True,
                        activation="relu", dropout=0.0, weight_init="gaussian", depthwise_init="current", zero_bias_init=True,
                        cfg_coef=3.0)
    chroma_proj = torch.nn.Linear(12, DIM)
    for module in (lm_melody, chroma_proj):
        with torch.no_grad():
            for p in module.parameters():
                p.copy_(p.float())
        module.double().eval()
    with torch.no_grad():
        chroma_embeds = chroma_proj(chroma)
        chroma_mask = torch.ones(1, chroma.shape[1], dtype=torch.long)
        conditions = {"self_wav": (chroma_embeds, chroma_mask), "description": (embeds, mask)}
        out = lm_melody.compute_predictions(codes, [], condition_tensors=conditions)
        melody_logits, melody_valid = out.logits, out.mask
        melody_ce = torch.zeros([], dtype=torch.float64)
        for k in range(K):
            m = melody_valid[0, k]
            melody_ce += torch.nn.functional.cross_entropy(melody_logits[0, k][m], codes[0, k][m])
        melody_ce = melody_ce / K
        cfg = {name: (torch.cat([c, torch.zeros_like(c)]), torch.cat([m, torch.zeros_like(m)])) for name, (c, m) in conditions.items()}
        gen = torch.full((1, K, GENERATED), -1, dtype=torch.long)
        seq, _, smask = pattern.build_pattern_sequence(gen, lm_melody.special_token_id)
        with lm_melody.streaming():
            uncond_state = lm_melody.get_streaming_state()
            prev = 0
            for offset in range(start, seq.shape[-1]):
                nxt = lm_melody._sample_next_token(seq[..., prev:offset], cfg, uncond_state, use_sampling=False, cfg_coef=3.0)
                valid_step = smask[..., offset:offset + 1].expand(1, -1, -1)
                nxt[~valid_step] = lm_melody.special_token_id
                seq[..., offset:offset + 1] = torch.where(seq[..., offset:offset + 1] == -1, nxt, seq[..., offset:offset + 1])
                prev = offset
        melody_generated, _, _ = pattern.revert_pattern_sequence(seq, special_token=-1)
        # The prefix as the fuser builds it: each prepended condition goes in front, so description, then self_wav.
        prefix = torch.cat([embeds, chroma_embeds], dim=1)

    # ---- the stereo model ----
    stereo_delays = [0, 0, 1, 1, 2, 2, 3, 3]
    torch.manual_seed(77)
    lm_stereo = LMModel(DelayedPatternProvider(n_q=8, delays=stereo_delays), ConditioningProvider({}), fuser, n_q=8, card=CARD,
                        dim=DIM, num_heads=HEADS, num_layers=LAYERS, hidden_scale=4, norm="layer_norm", norm_first=True,
                        bias_proj=False, bias_ff=False, bias_attn=False, cross_attention=True, causal=True, custom=True,
                        activation="relu", dropout=0.0, weight_init="gaussian", depthwise_init="current", zero_bias_init=True,
                        cfg_coef=3.0)
    with torch.no_grad():
        for p in lm_stereo.parameters():
            p.copy_(p.float())
    lm_stereo.double().eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(8)
        stereo_codes = torch.randint(0, CARD, (1, 8, FRAMES), generator=g)
        out = lm_stereo.compute_predictions(stereo_codes, [], condition_tensors={"description": (embeds, mask)})
        stereo_logits, stereo_valid = out.logits, out.mask
        stereo_ce = torch.zeros([], dtype=torch.float64)
        for k in range(8):
            m = stereo_valid[0, k]
            stereo_ce += torch.nn.functional.cross_entropy(stereo_logits[0, k][m], stereo_codes[0, k][m])
        stereo_ce = stereo_ce / 8
        cfg = {"description": (torch.cat([embeds, torch.zeros_like(embeds)]), torch.cat([mask, torch.zeros_like(mask)]))}
        stereo_pattern = lm_stereo.pattern_provider.get_pattern(GENERATED)
        gen = torch.full((1, 8, GENERATED), -1, dtype=torch.long)
        seq, _, smask = stereo_pattern.build_pattern_sequence(gen, lm_stereo.special_token_id)
        with lm_stereo.streaming():
            uncond_state = lm_stereo.get_streaming_state()
            prev = 0
            for offset in range(stereo_pattern.get_first_step_with_timesteps(0), seq.shape[-1]):
                nxt = lm_stereo._sample_next_token(seq[..., prev:offset], cfg, uncond_state, use_sampling=False, cfg_coef=3.0)
                valid_step = smask[..., offset:offset + 1].expand(1, -1, -1)
                nxt[~valid_step] = lm_stereo.special_token_id
                seq[..., offset:offset + 1] = torch.where(seq[..., offset:offset + 1] == -1, nxt, seq[..., offset:offset + 1])
                prev = offset
        stereo_generated, _, _ = stereo_pattern.revert_pattern_sequence(seq, special_token=-1)

    def state(module, prefix=""):
        return {prefix + k: v.float().flatten().tolist() for k, v in module.state_dict().items()}

    return {
        "config": {"codebooks": K, "card": CARD, "dim": DIM, "heads": HEADS, "layers": LAYERS, "delays": DELAYS, "t5": T5},
        "lm": state(lm),
        "t5": state(t5),
        "projection": {"weight": proj.weight.float().flatten().tolist(), "bias": proj.bias.float().flatten().tolist()},
        "tokens": TOKENS,
        "condition": embeds[0].tolist(),
        "condition_float64_norm": exact[0].tolist(),
        "codes": codes[0].tolist(),
        "mask": valid[0].int().tolist(),
        "logits": torch.nan_to_num(logits[0], nan=0.0).tolist(),
        "loss": ce.item(),
        "generated": generated[0, :, :GENERATED].tolist(),
        "stereo": {
            "delays": stereo_delays,
            "lm": state(lm_stereo),
            "codes": stereo_codes[0].tolist(),
            "mask": stereo_valid[0].int().tolist(),
            "logits": torch.nan_to_num(stereo_logits[0], nan=0.0).tolist(),
            "loss": stereo_ce.item(),
            "generated": stereo_generated[0, :, :GENERATED].tolist(),
        },
        "melody": {
            "tone": {"sample_rate": sr, "samples": n, "frequencies": [261.63, 440.0], "amplitude": 0.5},
            "chroma": chroma[0].tolist(),
            "lm": state(lm_melody),
            "projection": {"weight": chroma_proj.weight.float().flatten().tolist(), "bias": chroma_proj.bias.float().flatten().tolist()},
            "prefix": prefix[0].tolist(),
            "mask": melody_valid[0].int().tolist(),
            "logits": torch.nan_to_num(melody_logits[0], nan=0.0).tolist(),
            "loss": melody_ce.item(),
            "generated": melody_generated[0, :, :GENERATED].tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--audiocraft", default=os.path.expanduser("~/source/repos/ref/audiocraft"))
    args = parser.parse_args()
    result = compute(args.audiocraft)
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as f:
            json.dump(result, f)
        print(f"wrote loss {result['loss']:.6f}, generated {result['generated']}")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = (abs(stored["loss"] - result["loss"]) < 1e-12 and stored["generated"] == result["generated"]
            and abs(stored["melody"]["loss"] - result["melody"]["loss"]) < 1e-12
            and stored["melody"]["generated"] == result["melody"]["generated"]
            and abs(stored["stereo"]["loss"] - result["stereo"]["loss"]) < 1e-12
            and stored["stereo"]["generated"] == result["stereo"]["generated"])
    print("match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
