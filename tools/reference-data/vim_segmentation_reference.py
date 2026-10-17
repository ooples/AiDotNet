"""Vision Mamba segmentation reference: the official VisionMambaSeg (hustvl/Vim seg/backbone/vim.py, at the commit
vim_reference.py pins) with mmsegmentation 0.30's own UPerHead and FCNHead, as seg/configs/_base_/models/upernet_vim.py
assembles them.

The backbone runs exactly as in vim_reference.py (official code, the repository's pure-PyTorch reference kernels,
float64). mmcv's compiled ops are not needed by these heads, so mmcv.ops is a stub. VisionMambaSeg passes its
in_chans and num_classes positionally into VisionMamba's d_state and channels slots at this commit, so the intended
state size and channel count are passed through those two arguments. A small model (64x64 images,
16x16 patches, width 16, 4 layers, state 4, features from every layer, no class token, RMSNorm, the BEiT-style
multi-level neck, a 16-channel UPerHead with pyramid pooling scales 1, 2, 3, 6 and an FCN auxiliary head on the third
feature, 3 classes, all BatchNorm and dropout in eval mode) runs on two images, once as upernet_vim_tiny builds it
and once with upernet_vim_small's 2D rotary embedding (on the hidden states and the residual). The fixture records the state dict
(BatchNorm running statistics randomized), the images, random labels, the decode and auxiliary logits resized to the
image (EncoderDecoder), the loss CE(decode) + 0.4 CE(auxiliary), and that loss after one plain SGD step (lr 0.1).
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/vim_segmentation_reference.json.
"""
import argparse
import json
import os
import sys
import types
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vim_reference  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/vim_segmentation_reference.json"
SEG = f"https://raw.githubusercontent.com/hustvl/Vim/{vim_reference.COMMIT}/seg/backbone/vim.py"


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None


def compute():
    import tempfile
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)
    vim_reference.install(vim_reference.fetch())
    to_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self if self.dtype == torch.float64 else (
        self.double() if not self.is_floating_point() else to_float(self, *a, **k))

    sys.modules["mmcv.ops"] = _Stub("mmcv.ops")
    from mmseg.models.decode_heads.fcn_head import FCNHead
    from mmseg.models.decode_heads.uper_head import UPerHead
    vim_reference.module("mmcv_custom", load_checkpoint=None)
    vim_reference.module("utils", interpolate_pos_embed=None)
    path = os.path.join(tempfile.gettempdir(), "aidotnet-vim-" + vim_reference.COMMIT[:12], "seg_vim.py")
    if not os.path.exists(path):
        urllib.request.urlretrieve(SEG, path)
    seg = vim_reference.load("seg_vim", path)

    def case(name, rope, seed):
        torch.manual_seed(seed)
        # VisionMambaSeg forwards (img_size, patch_size, stride, depth, embed_dim, in_chans, num_classes) positionally to
        # VisionMamba(img_size, patch_size, stride, depth, embed_dim, d_state, channels, num_classes), so at this commit
        # its in_chans lands in d_state and its num_classes in channels. The intended values go through those slots.
        # The RoPE case follows upernet_vim_small: rotary embedding on the hidden states and the residual, residual in
        # fp32, and a class token that VisionMambaSeg removes (with its position-table row).
        backbone = seg.VisionMambaSeg(img_size=64, patch_size=16, in_chans=4, num_classes=3, embed_dim=16, depth=4,
                                      out_indices=[0, 1, 2, 3], pretrained=None, rms_norm=True,
                                      residual_in_fp32=rope, fused_add_norm=True, if_abs_pos_embed=True, if_rope=rope,
                                      if_rope_residual=rope, bimamba_type="v2", final_pool_type="all",
                                      if_divide_out=True, if_cls_token=rope, drop_path_rate=0.0)
        norm = dict(type="BN", requires_grad=True)
        decode = UPerHead(in_channels=[16, 16, 16, 16], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=16,
                          dropout_ratio=0.1, num_classes=3, norm_cfg=norm, align_corners=False)
        auxiliary = FCNHead(in_channels=16, in_index=2, channels=16, num_convs=1, concat_input=False,
                            dropout_ratio=0.1, num_classes=3, norm_cfg=norm, align_corners=False)
        model = torch.nn.ModuleDict({"backbone": backbone, "decode_head": decode, "auxiliary_head": auxiliary}).double()
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.float32).double())
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    c = m.num_features
                    m.weight.copy_((1 + 0.2 * torch.randn(c, generator=g)).to(torch.float32).double())
                    m.bias.copy_((0.1 * torch.randn(c, generator=g)).to(torch.float32).double())
                    m.running_mean.copy_((0.1 * torch.randn(c, generator=g)).to(torch.float32).double())
                    m.running_var.copy_((1 + 0.5 * torch.rand(c, generator=g)).to(torch.float32).double())
        state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()
                 if "num_batches" not in k and "rope.freqs" not in k}
        images = torch.randn(2, 3, 64, 64, generator=g).to(torch.float32).double()
        labels = torch.randint(0, 3, (2, 64, 64), generator=g)
        model.eval()

        def run():
            features = backbone(images)
            decode_logits = F.interpolate(decode(features), size=(64, 64), mode="bilinear", align_corners=False)
            auxiliary_logits = F.interpolate(auxiliary(features), size=(64, 64), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(decode_logits, labels) + 0.4 * F.cross_entropy(auxiliary_logits, labels)
            return features, decode_logits, auxiliary_logits, loss

        features, decode_logits, auxiliary_logits, loss = run()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.1 * p.grad
            after = run()[3].item()
        return {"name": name, "rope": rope, "state": state, "images": images.flatten().tolist(),
                "labels": labels.flatten().tolist(), "feature_shapes": [list(f.shape) for f in features],
                "features": [f.detach().flatten().tolist() for f in features],
                "decode_logits": decode_logits.detach().flatten().tolist(),
                "auxiliary_logits": auxiliary_logits.detach().flatten().tolist(), "loss": loss.item(),
                "loss_after_sgd": after}

    return {"commit": vim_reference.COMMIT, "cases": [case("plain", False, 3), case("rope", True, 5)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", [(c["name"], round(c["loss"], 9), round(c["loss_after_sgd"], 9)) for c in result["cases"]])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a["loss"] - b["loss"]) < 1e-12 and abs(a["loss_after_sgd"] - b["loss_after_sgd"]) < 1e-12
               for a, b in zip(stored["cases"], result["cases"]))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
