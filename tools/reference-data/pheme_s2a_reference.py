"""Pheme acoustic-model reference: Pheme's own TTSConformer (modules/s2a_model.py).

Builds a tiny randomly initialized TTSConformer with Pheme's code (PolyAI-LDN/pheme; pass a checkout with --pheme, the
code is not vendored here), stores its weights in float32 under the Lightning module's "model." prefix (the layout of
s2a.ckpt's state_dict), and records, in float64, the logits of one training-path forward pass per codebook level (the
model's own seeded mask after a 4-frame prompt) for fixed acoustic codes, semantic codes and an L2-normalized speaker
embedding, with the masked positions, in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/pheme_s2a_reference.json.

The speaker-embedding dropout is 0: the reference calls F.dropout without the training flag, so any other rate makes
the forward pass random. ChanLayerNorm keeps its float32 epsilon (see conformer_reference.py). The attention uses the
einsum path (attn_flash=False), the same arithmetic as the flash kernel. Default: verify; --write: regenerate.
"""
import argparse
import base64
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/pheme_s2a_reference.json"
HP = dict(hidden_size=16, enc_nlayers=1, nheads=2, depthwise_conv_kernel_size=5, dropout=0.0, n_codes=16,
          n_semantic_codes=16, n_cluster_groups=2, use_spkr_emb=True, speaker_embed_dropout=0.0)
TIME = 9
START = 4
MASK_RATIO = 0.6


def _chan_layer_norm_float32_epsilon(self, x):
    var = torch.var(x, dim=1, unbiased=False, keepdim=True)
    mean = torch.mean(x, dim=1, keepdim=True)
    return (x - mean) * var.clamp(min=1e-6).rsqrt() * self.gamma


def build(s2a, state=None):
    torch.manual_seed(2294)
    model = s2a.TTSConformer(SimpleNamespace(**HP))
    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = False
    if state is not None:
        model.load_state_dict({k[len("model."):]: v for k, v in state.items()}, strict=True)
    else:
        # Pheme.init_weights draws N(0, 0.02); larger draws make the comparison less trivially near zero.
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p) * 0.3 + (1.0 if p.dim() == 1 and p.numel() == HP["hidden_size"] else 0.0))
            for e in list(model.embedding) + [model.semantic_embedding]:
                e.weight[HP["n_codes"]].zero_()
    return model.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    acoustic = torch.randint(0, HP["n_codes"], (1, TIME, HP["n_cluster_groups"]), generator=g)
    acoustic[0, 0, :] = HP["n_codes"] + 1                      # SPKR_1 front padding, as at inference
    acoustic[0, TIME - 1, 1] = HP["n_codes"]                   # one padding code
    semantic = torch.randint(0, HP["n_semantic_codes"], (1, TIME), generator=g)
    semantic[0, 0] = HP["n_semantic_codes"] + 1
    speaker = torch.randn(1, 512, generator=g, dtype=torch.float64)
    return acoustic, semantic, speaker / speaker.norm()


def logits(model, acoustic, semantic, speaker):
    # The training path: the model draws the mask itself (process_input zeroes the chosen positions only when no mask
    # is passed in), from a prompt prefix of START frames and a fixed mask ratio, under a fixed seed.
    out, masks = [], []
    with torch.no_grad():
        for level in range(HP["n_cluster_groups"]):
            torch.manual_seed(11 + level)
            y, mask, _, _ = model(acoustic, level, semantic, torch.tensor([TIME]), speaker_emb=speaker,
                                  mask_ratio=MASK_RATIO, start_t=START)
            out.append([float(v) for v in y.reshape(-1).tolist()])
            masks.append([int(t) for t in torch.nonzero(mask[0]).reshape(-1).tolist()])
    return out, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pheme", required=True, help="a checkout of PolyAI-LDN/pheme")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    sys.path.insert(0, args.pheme)
    from modules import conformer as conformer_module
    # s2a_model imports utils.load_checkpoint, which the published repository does not contain; it is only used to
    # load pretrained weights, which this script never does.
    import types
    sys.modules.setdefault("utils", types.ModuleType("utils")).load_checkpoint = None
    from modules import s2a_model as s2a
    conformer_module.ChanLayerNorm.forward = _chan_layer_norm_float32_epsilon

    acoustic, semantic, speaker = inputs()
    if args.write:
        model = build(s2a)
        state = {"model." + k: v.float().clone().contiguous() for k, v in model.state_dict().items()}
        model = build(s2a, state).double()
        values, masks = logits(model, acoustic, semantic, speaker)
        data = {"hp": HP, "time": TIME, "masked": masks,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "acoustic": acoustic[0].tolist(), "semantic": semantic[0].tolist(),
                "speaker": [float(v) for v in speaker.reshape(-1).tolist()],
                "logits": values}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(s2a, load_safetensors(base64.b64decode(data["safetensors_base64"]))).double()
    recomputed, masks = logits(model, acoustic, semantic, speaker)
    if masks != data["masked"]:
        sys.exit("MISMATCH: the masked positions differ")
    check = _fixture.Comparison("Pheme acoustic model")
    for level, (committed, again) in enumerate(zip(data["logits"], recomputed)):
        check.floats(f"logits level {level}", committed, again, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
