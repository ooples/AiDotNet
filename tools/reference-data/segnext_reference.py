"""Reference fixture for SegNeXt (Guo et al. 2022) from the official code, in float64.

Runs mmsegmentation's own source -- mmseg/models/backbones/mscan.py and mmseg/models/decode_heads/ham_head.py,
downloaded at the pinned commit below -- with minimal stand-ins for the mmcv/mmengine helpers they import
(ConvModule as conv -> norm -> ReLU with mmcv's bias='auto' and 'bn'/'gn' names, build_norm_layer, DropPath,
BaseModule, resize). BaseDecodeHead is reduced to what LightHamHead uses: 'multiple_select' inputs, Dropout2d and
the 1x1 conv_seg. The NMF's random bases are injected so both sides use the same draw.

A small MSCAN (embed 8/16/32/32, depths 1/1/2/1, MLP ratio 2) and a LightHamHead (32 channels, GN(32), rank 4,
3 classes) on two 64x64 images with random weights and BatchNorm statistics. Records the four stage outputs, the
eval-mode logits (7 NMF steps), the mean pixel cross-entropy after resizing the logits to the labels (mmseg's
loss_by_feat), the loss after one SGD step of 0.1 on that loss, and the train-mode logits (BatchNorm batch
statistics, 6 NMF steps; drop path and dropout 0).

Writes tests/AiDotNet.Tests/ComputerVision/ReferenceData/segnext_reference.json. Run with the reference venv.
"""
import json
import math
import os
import types
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
COMMIT = "b040e147adfa027bbc071b624bedf0ae84dfc922"  # mmsegmentation main when this fixture was generated
BASE = f"https://raw.githubusercontent.com/open-mmlab/mmsegmentation/{COMMIT}/"


def fetch(path):
    with urllib.request.urlopen(BASE + path) as r:
        return r.read().decode("utf-8")


# ---- stand-ins for the mmcv / mmengine helpers the official files import ----
def build_activation_layer(cfg):
    assert cfg["type"] in ("GELU", "ReLU")
    return nn.GELU() if cfg["type"] == "GELU" else nn.ReLU()


def build_norm_layer(cfg, channels):
    if cfg["type"] in ("BN", "SyncBN"):
        return "bn", nn.BatchNorm2d(channels)
    if cfg["type"] == "GN":
        return "gn", nn.GroupNorm(cfg["num_groups"], channels)
    raise ValueError(cfg)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        assert not self.training or self.p == 0
        return x


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_cfg=None, norm_cfg=None, act_cfg=dict(type="ReLU")):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=norm_cfg is None)
        self.norm_name = None
        if norm_cfg is not None:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        self.act = nn.ReLU() if act_cfg is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm_name:
            x = getattr(self, self.norm_name)(x)
        return self.act(x) if self.act is not None else x


def resize(x, size, mode, align_corners):
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class Registry:
    def register_module(self):
        return lambda cls: cls


class BaseDecodeHead(nn.Module):
    """The part of mmseg's BaseDecodeHead LightHamHead uses."""

    def __init__(self, in_channels, channels, *, num_classes, input_transform, in_index, dropout_ratio=0.1,
                 norm_cfg=None, act_cfg=dict(type="ReLU"), align_corners=False, conv_cfg=None, **kwargs):
        super().__init__()
        assert input_transform == "multiple_select"
        self.in_channels, self.channels, self.in_index = in_channels, channels, in_index
        self.norm_cfg, self.act_cfg, self.conv_cfg, self.align_corners = norm_cfg, act_cfg, conv_cfg, align_corners
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def _transform_inputs(self, inputs):
        return [inputs[i] for i in self.in_index]

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)


def load_official(path, replacements):
    source = fetch(path)
    for old, new in replacements:
        assert old in source, (path, old)
        source = source.replace(old, new)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, warnings=__import__("warnings"),
                     build_activation_layer=build_activation_layer, build_norm_layer=build_norm_layer,
                     DropPath=DropPath, BaseModule=BaseModule, ConvModule=ConvModule, resize=resize,
                     BaseDecodeHead=BaseDecodeHead, MODELS=Registry(), get_device=lambda: "cpu",
                     constant_init=None, normal_init=None, trunc_normal_init=None)
    exec(compile(source, path, "exec"), namespace)
    return namespace


mscan = load_official("mmseg/models/backbones/mscan.py", [
    ("from mmcv.cnn import build_activation_layer, build_norm_layer\n", ""),
    ("from mmcv.cnn.bricks import DropPath\n", ""),
    ("from mmengine.model import BaseModule\n", ""),
    ("from mmengine.model.weight_init import (constant_init, normal_init,\n                                        trunc_normal_init)\n", ""),
    ("from mmseg.registry import MODELS\n", ""),
])
ham = load_official("mmseg/models/decode_heads/ham_head.py", [
    ("from mmcv.cnn import ConvModule\n", ""),
    ("from mmengine.device import get_device\n", ""),
    ("from mmseg.registry import MODELS\n", ""),
    ("from ..utils import resize\n", ""),
    ("from .decode_head import BaseDecodeHead\n", ""),
])

EMBED, DEPTHS, MLP = [8, 16, 32, 32], [1, 1, 2, 1], [2, 2, 2, 2]
CHANNELS, RANK, CLASSES, SIZE = 32, 4, 3, 64


def build():
    backbone = mscan["MSCAN"](embed_dims=EMBED, mlp_ratios=MLP, drop_rate=0.0, drop_path_rate=0.0, depths=DEPTHS,
                              norm_cfg=dict(type="BN", requires_grad=True))
    head = ham["LightHamHead"](in_channels=EMBED[1:], in_index=[1, 2, 3], channels=CHANNELS, ham_channels=CHANNELS,
                               dropout_ratio=0.0, num_classes=CLASSES, norm_cfg=dict(type="GN", num_groups=32),
                               align_corners=False, ham_kwargs=dict(MD_S=1, MD_R=RANK, train_steps=6, eval_steps=7,
                                                                    inv_t=100, rand_init=True))
    return backbone, head


def main():
    gen = torch.Generator().manual_seed(7)
    backbone, head = build()
    with torch.no_grad():
        for module in list(backbone.modules()) + list(head.modules()):
            for name, p in module.named_parameters(recurse=False):
                if name == "weight" and isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=gen))
                elif name.startswith("layer_scale"):
                    p.copy_(0.5 + 0.2 * torch.randn(p.shape, generator=gen))
                else:
                    fan_in = p[0].numel() if p.dim() > 1 else 1
                    p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(fan_in) * (1.0 if p.dim() > 1 else 0.2))
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.copy_(0.1 * torch.randn(module.num_features, generator=gen))
                module.running_var.copy_(0.5 + torch.rand(module.num_features, generator=gen))
    # Cloned: state_dict() shares storage with the parameters, which the SGD step below changes in place.
    state = {f"backbone.{k}": v.clone() for k, v in backbone.state_dict().items() if not k.endswith("num_batches_tracked")}
    state.update({f"decode_head.{k}": v.clone() for k, v in head.state_dict().items() if not k.endswith("num_batches_tracked")})

    images = torch.randn(2, 3, SIZE, SIZE, generator=gen)
    labels = torch.randint(0, CLASSES, (2, SIZE, SIZE), generator=gen)
    hw = (SIZE // 8) * (SIZE // 8)
    eval_bases = F.normalize(torch.rand(2, CHANNELS, RANK, generator=gen), dim=1)
    train_bases = F.normalize(torch.rand(2, CHANNELS, RANK, generator=gen), dim=1)
    nmf = head.hamburger.ham
    use = {"bases": eval_bases}
    nmf._build_bases = lambda B, S, D, R, device=None: use["bases"].clone()

    def run(train):
        backbone.train(train)
        head.train(train)
        feats = backbone(images)
        logits = head(feats)
        up = resize(logits, labels.shape[1:], "bilinear", False)
        return feats, logits, F.cross_entropy(up, labels)

    feats, logits, loss = run(False)
    params = list(backbone.parameters()) + list(head.parameters())
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= 0.1 * g
        _, _, after = run(False)
        use["bases"] = train_bases
        _, train_logits, _ = run(True)

    out = {
        "state": {k: v.flatten().tolist() for k, v in state.items()},
        "shapes": {k: list(v.shape) for k, v in state.items()},
        "images": images.flatten().tolist(),
        "labels": labels.flatten().tolist(),
        "eval_bases": eval_bases.flatten().tolist(),
        "train_bases": train_bases.flatten().tolist(),
        "features": [f.flatten().tolist() for f in feats],
        "feature_shapes": [list(f.shape) for f in feats],
        "logits": logits.flatten().tolist(),
        "loss": loss.item(),
        "loss_after_sgd": after.item(),
        "train_logits": train_logits.flatten().tolist(),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "AiDotNet.Tests", "ComputerVision", "ReferenceData",
                        "segnext_reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("wrote", os.path.normpath(path), "loss", loss.item(), "after", after.item(), "hw", hw)


if __name__ == "__main__":
    main()
