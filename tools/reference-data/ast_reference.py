"""Audio Spectrogram Transformer reference: the official code (YuanGongND/ast, src/models/ast_models.py at
31088be8a3f6ef96416145c4b8d43c81f99eba7a) and torchaudio's Kaldi filterbank.

The official ASTModel builds its body with timm 0.4.5's DeiT (distilled ViT) and replaces the patch embedding with a
1-channel convolution of stride (fstride, tstride). timm 0.4.5 does not install on current Python, so the body here is
timm's VisionTransformerDistilled (the same blocks: pre-norm LayerNorm eps 1e-6, fused qkv with bias, exact GELU MLP),
and the official PatchEmbed, __init__ (imagenet_pretrain=False and the ImageNet adaptation) and forward are copied below
with only the timm.create_model call changed to a small model.

Recorded:
  * two small networks (width 48, 4 heads, 2 blocks, 5 labels) on [2, frames, mel] filterbanks, one with the paper's
    10x10 strides on a 32-bin, 48-frame input, one with 8 (frequency) x 12 (time) strides on 40 bins and 60 frames:
    the state dict, the input, the pooled representation, the logits, the BCE-with-logits loss against fixed targets,
    and the loss after one plain SGD step (lr 0.1) on every parameter;
  * Kaldi fbank (torchaudio.compliance.kaldi.fbank as the official dataloader calls it: htk_compat, no energy,
    25 ms frames every 10 ms, no dither, 128 bins) of a 1.05 s 16 kHz waveform with the Hanning window the code uses and
    the Hamming window the paper states;
  * the ImageNet adaptation (paper 2.2) of a small DeiT (64-pixel images, a 4x4 patch grid) to a 2x6 AST grid: the
    frequency side is cut, the time side interpolated. The paper averages the three input channels of the patch
    projection where the code sums them; the fixture averages.
Default: verify; --write: regenerate tests/AiDotNet.Tests/Audio/ReferenceData/ast_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/ast_reference.json"


def official_classes():
    import torch
    import torch.nn as nn
    from timm.models.deit import VisionTransformerDistilled
    from timm.layers import to_2tuple, trunc_normal_

    class PatchEmbed(nn.Module):  # official
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
            super().__init__()
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_patches = num_patches
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            return x

    def deit(img_size, embed_dim, depth, heads):
        v = VisionTransformerDistilled(img_size=img_size, patch_size=16, embed_dim=embed_dim, depth=depth,
                                       num_heads=heads, mlp_ratio=4, qkv_bias=True, num_classes=10)
        assert v.norm.eps == 1e-6 and v.blocks[0].norm1.eps == 1e-6
        v.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=embed_dim)
        return v

    class ASTModel(nn.Module):  # official, imagenet_pretrain=False branch; timm.create_model replaced by deit(...)
        def __init__(self, label_dim, fstride, tstride, input_fdim, input_tdim, embed_dim, depth, heads):
            super().__init__()
            self.v = deit(64, embed_dim, depth, heads)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
            test_input = torch.randn(1, 1, input_fdim, input_tdim)
            test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            test_out = test_proj(test_input)
            f_dim = test_out.shape[2]
            t_dim = test_out.shape[3]
            return f_dim, t_dim

        def representation(self, x):
            x = x.unsqueeze(1)
            x = x.transpose(2, 3)
            B = x.shape[0]
            x = self.v.patch_embed(x)
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
            x = x + self.v.pos_embed
            x = self.v.pos_drop(x)
            for blk in self.v.blocks:
                x = blk(x)
            x = self.v.norm(x)
            x = (x[:, 0] + x[:, 1]) / 2
            return x

        def forward(self, x):
            return self.mlp_head(self.representation(x))

    def adapt(deit_model, f_dim, t_dim, embed_dim):
        """The official ImageNet branch (pos_embed cut / interpolate) with the paper's channel average."""
        v = deit_model
        original_num_patches = v.patch_embed.num_patches
        oringal_hw = int(original_num_patches ** 0.5)
        original_embedding_dim = embed_dim
        num_patches = f_dim * t_dim
        proj_weight = torch.mean(v.patch_embed.proj.weight, dim=1).unsqueeze(1)
        new_pos_embed = v.pos_embed[:, 2:, :].detach().reshape(1, original_num_patches, original_embedding_dim).transpose(1, 2).reshape(1, original_embedding_dim, oringal_hw, oringal_hw)
        if t_dim <= oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(oringal_hw / 2) - int(t_dim / 2): int(oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(oringal_hw, t_dim), mode='bilinear')
        if f_dim <= oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(oringal_hw / 2) - int(f_dim / 2): int(oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.reshape(1, original_embedding_dim, num_patches).transpose(1, 2)
        pos = torch.cat([v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
        return proj_weight, pos

    return ASTModel, deit, adapt


CASES = [
    {"name": "paper strides", "fstride": 10, "tstride": 10, "mel": 32, "frames": 48},
    {"name": "rectangular strides", "fstride": 8, "tstride": 12, "mel": 40, "frames": 60},
]


def network_case(case, seed):
    import torch
    ASTModel, _, _ = official_classes()
    torch.manual_seed(seed)
    model = ASTModel(label_dim=5, fstride=case["fstride"], tstride=case["tstride"], input_fdim=case["mel"],
                     input_tdim=case["frames"], embed_dim=48, depth=2, heads=4).double()
    with torch.no_grad():  # zero-initialized tokens would hide a token-order mistake
        model.v.cls_token.normal_(0, 0.5)
        model.v.dist_token.normal_(0, 0.5)
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()
             if not k.startswith("v.head")}
    x = torch.randn(2, case["frames"], case["mel"], dtype=torch.float64)
    targets = (torch.rand(2, 5, dtype=torch.float64) > 0.5).double()
    representation = model.representation(x)
    logits = model.mlp_head(representation)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
        after = torch.nn.functional.binary_cross_entropy_with_logits(model(x), targets).item()
    return {"name": case["name"], "fstride": case["fstride"], "tstride": case["tstride"], "mel": case["mel"],
            "frames": case["frames"], "state": state, "input": x.flatten().tolist(),
            "targets": targets.flatten().tolist(), "representation": representation.detach().flatten().tolist(),
            "logits": logits.detach().flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after}


def fbank_case():
    import torch
    import torchaudio
    torch.manual_seed(11)
    t = torch.arange(16800, dtype=torch.float64) / 16000
    waveform = (0.3 * torch.sin(2 * torch.pi * 440 * t) + 0.05 * torch.randn(16800, dtype=torch.float64)).unsqueeze(0)
    waveform = waveform - waveform.mean()
    out = {"waveform": waveform.flatten().tolist()}
    for window in ("hanning", "hamming"):
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                  window_type=window, num_mel_bins=128, dither=0.0, frame_shift=10)
        out[window] = fbank.flatten().tolist()
        out["frames"] = fbank.shape[0]
    return out


def adaptation_case():
    import torch
    _, deit, adapt = official_classes()
    torch.manual_seed(5)
    v = deit(64, 48, 2, 4).double()
    with torch.no_grad():
        for p in v.parameters():
            p.normal_(0, 0.5)
    state = {k: p.detach().flatten().tolist() for k, p in v.state_dict().items() if not k.startswith("head")}
    proj, pos = adapt(v, 2, 6, 48)
    return {"grid": 4, "f_dim": 2, "t_dim": 6, "state": state, "proj": proj.detach().flatten().tolist(),
            "pos_embed": pos.detach().flatten().tolist()}


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    return {"networks": [network_case(c, 21 + i) for i, c in enumerate(CASES)], "fbank": fbank_case(),
            "adaptation": adaptation_case()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    losses = [(n["loss"], n["loss_after_sgd"]) for n in result["networks"]]
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", losses)
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a - b) < 1e-12 for (x, y), n in zip(losses, stored["networks"])
               for a, b in ((x, n["loss"]), (y, n["loss_after_sgd"])))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
