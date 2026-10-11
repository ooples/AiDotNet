"""MusicGen reference: audiocraft's LMModel (models/lm.py) with the delay pattern and text cross-attention.

A tiny configuration of the reference language model (4 codebooks of 16 codes, D = 16, 2 heads, 2 layers, the
MusicGen settings: pre-norm, bias-free linears, cross-attention, and the paper's ReLU feed-forward) is conditioned on
a tiny Hugging Face T5 encoder and audiocraft's text projection (T5Conditioner.output_proj). The fixture records:
- the weights (LM state dict, T5 encoder state dict, output_proj), float32;
- the conditioning of a token sequence, the logits and validity mask of compute_predictions for a code sequence,
  and the training cross-entropy (MusicGenSolver._compute_cross_entropy);
- the conditioning again with T5LayerNorm's mean square taken in float64: Hugging Face computes it in float32 even
  in a float64 model, which puts ~1e-7 of rounding into the stock conditioning;
- codes generated greedily with classifier-free guidance 3.0, following LMModel.generate's loop.
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
    same = abs(stored["loss"] - result["loss"]) < 1e-12 and stored["generated"] == result["generated"]
    print("match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
