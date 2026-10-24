"""GLaMM region encoder reference: the official MLVLROIQueryModule (mbzuai-oryx/groundingLMM, model/layers.py),
transcribed with its mmcv and mmdet pieces replaced by their exact torch equivalents, run in float64 on CPU.

The official module needs mmcv's ConvModule and RoIAlign and mmdet's BaseRoIExtractor. ConvModule(3x3, GN(64), ReLU)
is a bias-free convolution (bias='auto' with a norm), GroupNorm and ReLU; mmcv's RoIAlign(output 14, sampling ratio 2,
aligned=True) computes exactly torchvision.ops.roi_align(..., aligned=True); BaseRoIExtractor contributes one RoIAlign per
level with spatial scale 1/stride. Everything else is the official code as written, including its quirks: the boxes are
scaled by 224 although CLIP-L/14-336's levels map a 336 px image, and the fused levels come from the shuffle below. The
official code fixes the region and position-embedding width at 1024 and the output at 4096 (the LLM width); they are the
REGION and OUT parameters here, so the fixture stays small (the C# defaults are the official values). The official code
casts the upsampled levels to bfloat16 (a mixed-precision artifact); that cast is left out here.

A small configuration keeps the module's structure: four levels of 64 channels on a 6x6 grid (upsampled to 48, 24, 12 and
6), GroupNorm with 64 groups as written, five fuse rounds and a 14x14 RoIAlign, a region width of 32 and an output of 48.
For two images with two and one boxes the fixture records the region features, their
mean square, and that loss after one SGD step (lr 0.01). The weights and features are not stored: both sides compute them
from the parameter names (synthetic(), as in east_reference.py).
Default: verify; --write: regenerate tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_region_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_region_reference.json"
EMBED = 64
REGION = 32
OUT = 48
GRID = 6
BOXES = [[[0.10, 0.20, 0.55, 0.70], [0.40, 0.05, 0.95, 0.60]], [[0.25, 0.30, 0.80, 0.90]]]


def build():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.ops import roi_align

    class ConvModule(nn.Sequential):  # mmcv ConvModule(3x3, padding 1, norm_cfg=GN(64)): conv without bias, GN, ReLU
        def __init__(self, channels):
            super().__init__(nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False), nn.GroupNorm(64, channels),
                             nn.ReLU(inplace=False))

    class MLVLFuseModule(nn.Module):
        def __init__(self, input_dims=1024, embed_dims=1024, num_levels=3, num_fuse=4):
            super().__init__()
            self.embed_dims = embed_dims
            self.num_levels = num_levels
            self.num_fuse = num_fuse
            self.input_dims = input_dims
            self.shuffle_channles = embed_dims // 4
            self.fuse_lvl_list = []
            for lvl in range(num_levels):
                top_lvl = min(lvl + 1, num_levels - 1)
                dow_lvl = max(lvl - 1, 0)
                self.fuse_lvl_list.append((lvl, top_lvl, dow_lvl))
            self.remain_chs = self.embed_dims - self.shuffle_channles * 2
            self.input_conv = nn.ModuleList([nn.Conv2d(self.input_dims + 2, self.embed_dims, 1) for _ in range(num_levels)])
            self.fuse_convs = nn.ModuleList(ConvModule(self.embed_dims) for _ in range(num_fuse))

        def generate_coordinate(self, featmap_sizes):
            x_range = torch.linspace(-1, 1, featmap_sizes[-1])
            y_range = torch.linspace(-1, 1, featmap_sizes[-2])
            y, x = torch.meshgrid(y_range, x_range, indexing="ij")
            y = y.expand([featmap_sizes[0], 1, -1, -1])
            x = x.expand([featmap_sizes[0], 1, -1, -1])
            return torch.cat([x, y], 1)

        def _single_shuffle(self, inputs, conv_module):
            for single_conv_m in conv_module:
                fused_inputs = []
                for tar_lvl, top_lvl, dow_lvl in self.fuse_lvl_list:
                    tar_input, top_input, down_input = inputs[tar_lvl], inputs[top_lvl], inputs[dow_lvl]
                    remain = tar_input[:, :self.remain_chs]
                    from_top = top_input[:, self.remain_chs:][:, self.shuffle_channles:]
                    from_top = F.interpolate(from_top, size=tar_input.shape[-2:], mode="bilinear", align_corners=True)
                    from_down = down_input[:, self.remain_chs:][:, :self.shuffle_channles]
                    from_down = F.interpolate(from_down, size=tar_input.shape[-2:], mode="bilinear", align_corners=True)
                    fused_inputs.append(torch.cat([remain, from_top, from_down], dim=1))
                inputs = [single_conv_m(item) for item in fused_inputs]
            return inputs

        def forward(self, inputs):
            inputs = [torch.cat([feat, self.generate_coordinate(feat.shape)], dim=1) for feat in inputs]
            inputs = [self.input_conv[lvl](item) for lvl, item in enumerate(inputs)]
            for conv_m in self.fuse_convs:
                inputs = self._single_shuffle(inputs, [conv_m])
            return inputs

    class MlvlRoIExtractor(nn.Module):
        def __init__(self, out_channels, featmap_strides, embed_dims=1024, fuse_level=3):
            super().__init__()
            self.embed_dims = embed_dims
            self.fuse_level = fuse_level
            self.out_channels = out_channels
            self.featmap_strides = featmap_strides
            self.output_size = 14
            self.pconvs = nn.ModuleList(nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1) for _ in range(fuse_level))
            self.pos_embedd = nn.Sequential(nn.Linear(4, 256), nn.ReLU(inplace=False), nn.LayerNorm(256), nn.Linear(256, REGION),
                                            nn.ReLU(inplace=False), nn.LayerNorm(REGION))
            self.updims = nn.Linear(REGION, OUT)
            self.flatten_linear = nn.Linear(self.embed_dims * self.output_size ** 2, REGION)
            for m in self.modules():  # norm_init_weights: normal_init(m, 0, 0.01) on every Conv2d
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def forward(self, feats, rois):
            num_imgs = len(rois)
            batch_rois = torch.cat(rois, dim=0)
            pos_embedd = self.pos_embedd(batch_rois)
            new_rois = []
            for img_id, single_img_roi in enumerate(rois):
                single_img_roi = single_img_roi * 224
                roi_img_id = single_img_roi.new_ones(len(single_img_roi)) * img_id
                new_rois.append(torch.cat([roi_img_id[:, None], single_img_roi], dim=1))
            rois = torch.cat(new_rois)
            roi_feats = []
            for i in range(len(feats)):
                roi_feats.append(roi_align(feats[i], rois, output_size=self.output_size, spatial_scale=1.0 / self.featmap_strides[i],
                                           sampling_ratio=2, aligned=True))
            fuse_roi_feats = sum(self.pconvs[i](roi_feats[i]) for i in range(self.fuse_level))
            fuse_roi_feats = F.relu(fuse_roi_feats).flatten(1, -1)
            fuse_roi_feats = self.flatten_linear(fuse_roi_feats) + pos_embedd
            fuse_roi_feats = self.updims(fuse_roi_feats)
            return [fuse_roi_feats[rois[:, 0] == i] for i in range(num_imgs)]

    class MLVLROIQueryModule(nn.Module):
        def __init__(self, embed_dims=1024, num_levels=3):
            super().__init__()
            self.mlvl_fuse = MLVLFuseModule(input_dims=embed_dims, embed_dims=embed_dims, num_levels=num_levels, num_fuse=5)
            strids = [14 / 8, 14 / 4, 14 / 2, 14]
            self.roi_align = MlvlRoIExtractor(out_channels=embed_dims, featmap_strides=strids, embed_dims=embed_dims,
                                              fuse_level=num_levels)

        def forward(self, mlvl_feats, bboxes):
            if mlvl_feats[0].dim() == 3:
                h = w = int(math.sqrt(mlvl_feats[0].shape[1]))
                b, c = mlvl_feats[0].shape[0], mlvl_feats[0].shape[-1]
                mlvl_feats = [item.reshape(b, h, w, c).permute(0, 3, 1, 2) for item in mlvl_feats]
            base_shape = mlvl_feats[0].shape[-2:]
            num_level = len(mlvl_feats)
            to_shape = [(base_shape[0] * 2 ** level, base_shape[1] * 2 ** level) for level in range(num_level)][::-1]
            mlvl_feats = [F.interpolate(mlvl_feats[level], size=to_shape[level], mode="bilinear", align_corners=True)
                          for level in range(num_level)]
            return self.roi_align(self.mlvl_fuse(mlvl_feats), bboxes)

    module = MLVLROIQueryModule(embed_dims=EMBED, num_levels=4).double()
    with torch.no_grad():  # deterministic weights both sides compute (no state stored)
        for name, p in module.named_parameters():
            p.copy_(torch.tensor(synthetic(name, list(p.shape))).reshape(p.shape))
    return module


def uniforms(name, count):
    """u in [0, 1) from FNV-1a of the name and a xorshift of the index; exact integer arithmetic (as in east_reference.py)."""
    h = 2166136261
    for ch in name.encode("ascii"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    out = []
    for j in range(count):
        x = (h ^ ((j * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out.append(x / 4294967296.0)
    return out


def synthetic(name, shape):
    """Norm weights near 1, small biases, and Kaiming-uniform weights (the module's own names)."""
    count = 1
    for d in shape:
        count *= d
    u = uniforms(name, count)
    if name.endswith(".bias"):
        return [0.1 * (2 * v - 1) for v in u]
    if (name.startswith("mlvl_fuse.fuse_convs.") and name.endswith(".1.weight")) or name in ("roi_align.pos_embedd.2.weight", "roi_align.pos_embedd.5.weight"):
        return [0.5 + v for v in u]
    bound = math.sqrt(6.0 / (count // shape[0]))
    return [bound * (2 * v - 1) for v in u]


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    module = build()
    feats = [torch.tensor([2 * v - 1 for v in uniforms(f"features{level}", 2 * GRID * GRID * EMBED)]).reshape(2, GRID * GRID, EMBED)
             for level in range(4)]
    boxes = [torch.tensor(b) for b in BOXES]

    def run():
        out = module(feats, boxes)
        return out, sum((o ** 2).sum() for o in out) / sum(o.numel() for o in out)

    out, loss = run()
    module.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in module.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        _, after = run()
    return {"boxes": BOXES,
            "regions": [o.detach().flatten().tolist() for o in out], "loss": loss.item(), "loss_after_sgd": after.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["loss"], result["loss_after_sgd"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(stored[k] - result[k]) < 1e-12 for k in ("loss", "loss_after_sgd"))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
