"""Vision Mamba (Vim) reference: the official code, hustvl/Vim at a pinned commit, run on the CPU.

The official VisionMamba (vim/models_mamba.py) and its Mamba mixer (mamba-1p1p1/mamba_ssm/modules/mamba_simple.py, with
bidirectional "v2" Mamba) are downloaded at commit dd0358ad1e42701f22afbefa0717cc8825cf9f45 and run unchanged. Only
their CUDA/Triton kernels are replaced, each by the pure-PyTorch reference the same repository ships:
- the selective scan by selective_scan_ref (mamba_ssm/ops/selective_scan_interface.py);
- mamba_inner_fn_no_out_proj by mamba_inner_ref without its output projection (the same file): the causal depthwise
  convolution, SiLU, x_proj, dt_proj, and the selective scan gated by SiLU(z);
- the fused add + RMSNorm by rms_norm_ref (mamba_ssm/ops/triton/layer_norm.py).

A small Vim (16x16 images, 4x4 patches at stride 4, width 16, 2 layers, state 4, 5 classes, RMSNorm, residual in
fp32, middle class token, bimamba v2 with divided output, no drop path) runs in float64 from float32 weights on two
images (the official .float() casts, fp32 floors for half-precision training, are kept at float64). The fixture records the state dict, the images, the class-token features after the final norm, the logits,
the cross-entropy against targets (1, 3), and that loss after one plain SGD step (lr 0.1) on every parameter. A second
case uses 8x8 patches at stride 4 (overlapping patches, as the stride-8 Vim models do at their scale).
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/vim_reference.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
import urllib.request

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/vim_reference.json"
COMMIT = "dd0358ad1e42701f22afbefa0717cc8825cf9f45"
BASE = f"https://raw.githubusercontent.com/hustvl/Vim/{COMMIT}/"
FILES = {
    "models_mamba": "vim/models_mamba.py",
    "mamba_simple": "mamba-1p1p1/mamba_ssm/modules/mamba_simple.py",
    "selective_scan_interface": "mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py",
    "layer_norm": "mamba-1p1p1/mamba_ssm/ops/triton/layer_norm.py",
    "rope": "vim/rope.py",
}


def fetch():
    cache = os.path.join(tempfile.gettempdir(), "aidotnet-vim-" + COMMIT[:12])
    os.makedirs(cache, exist_ok=True)
    paths = {}
    for name, rel in FILES.items():
        path = os.path.join(cache, name + ".py")
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + rel, path)
        paths[name] = path
    return paths


def module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def install(paths):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange

    # Kernels that are absent on the CPU.
    module("selective_scan_cuda")
    module("causal_conv1d", causal_conv1d_fn=None, causal_conv1d_update=None)
    for pkg in ["mamba_ssm", "mamba_ssm.modules", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.utils"]:
        module(pkg).__path__ = []
    module("mamba_ssm.ops.triton.selective_state_update", selective_state_update=None)
    module("mamba_ssm.utils.generation", GenerationMixin=object)
    module("mamba_ssm.utils.hf", load_config_hf=None, load_state_dict_hf=None)
    load("rope", paths["rope"])

    ssi = load("mamba_ssm.ops.selective_scan_interface", paths["selective_scan_interface"])

    def mamba_inner_fn_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight, A, B=None, C=None,
                                   D=None, delta_bias=None, B_proj_bias=None, C_proj_bias=None, delta_softplus=True,
                                   checkpoint_lvl=1):
        # mamba_inner_ref without its output projection; causal_conv1d_fn(activation="silu") is the depthwise
        # convolution padded by width - 1 on the left, truncated to L, then SiLU.
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1]
        x, z = xz.chunk(2, dim=1)
        width = conv1d_weight.shape[-1]
        x = F.silu(F.conv1d(x, conv1d_weight, conv1d_bias, padding=width - 1, groups=x.shape[1])[..., :L])
        x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        B = rearrange(x_dbl[:, delta_rank:delta_rank + d_state], "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(x_dbl[:, -d_state:], "(b l) dstate -> b dstate l", l=L).contiguous()
        return ssi.selective_scan_ref(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)

    ssi.mamba_inner_fn_no_out_proj = mamba_inner_fn_no_out_proj
    ssi.selective_scan_fn = ssi.selective_scan_ref

    # The Triton file's pure-PyTorch reference norms, without importing Triton.
    source = open(paths["layer_norm"], encoding="utf-8").read()
    start, end = source.index("def layer_norm_ref("), source.index("def config_prune(")
    ln = module("mamba_ssm.ops.triton.layer_norm")
    exec(compile("import torch\nimport torch.nn.functional as F\n" + source[start:end], paths["layer_norm"], "exec"),
         ln.__dict__)

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.register_parameter("bias", None)

        def forward(self, x):
            return ln.rms_norm_ref(x, self.weight, self.bias, eps=self.eps)

    def rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6, **kwargs):
        return ln.rms_norm_ref(x, weight, bias, residual=residual, eps=eps, prenorm=prenorm, upcast=True)

    ln.RMSNorm, ln.rms_norm_fn, ln.layer_norm_fn = RMSNorm, rms_norm_fn, None
    load("mamba_ssm.modules.mamba_simple", paths["mamba_simple"])
    return load("models_mamba", paths["models_mamba"])


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    vim = install(fetch())
    # The official code calls .float() on the scan inputs, A, D and the delta bias, and the reference norm upcasts with
    # .float(): fp32 floors for half-precision training. In this float64 run they would be downcasts, so a float64
    # tensor stays float64 (the weights themselves are float32 values, so nothing else changes).
    to_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self if self.dtype == torch.float64 else (
        self.double() if not self.is_floating_point() else to_float(self, *a, **k))

    def case(name, patch, stride, seed):
        torch.manual_seed(seed)
        model = vim.VisionMamba(img_size=16, patch_size=patch, stride=stride, depth=2, embed_dim=16, d_state=4,
                                num_classes=5, drop_path_rate=0.0, rms_norm=True, residual_in_fp32=True,
                                fused_add_norm=True, final_pool_type="mean", if_abs_pos_embed=True, if_rope=False,
                                bimamba_type="v2", if_cls_token=True, if_divide_out=True,
                                use_middle_cls_token=True).double()
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.float())
        g = torch.Generator().manual_seed(seed + 1)
        images = torch.randn(2, 3, 16, 16, generator=g).float().double()
        targets = torch.tensor([1, 3])
        state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
        model.eval()
        features = model.forward_features(images)
        logits = model.head(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.1 * p.grad
            after = torch.nn.functional.cross_entropy(model(images), targets).item()
        return {"name": name, "patch": patch, "stride": stride, "state": state, "images": images.flatten().tolist(),
                "targets": targets.tolist(), "features": features.detach().flatten().tolist(),
                "logits": logits.detach().flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after,
                "tokens": model.patch_embed.num_patches + 1}

    return {"commit": COMMIT, "cases": [case("patch4", 4, 4, 3), case("patch8_stride4", 8, 4, 5)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", [(c["name"], c["tokens"], round(c["loss"], 9), round(c["loss_after_sgd"], 9)) for c in result["cases"]])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a["loss"] - b["loss"]) < 1e-12 and abs(a["loss_after_sgd"] - b["loss_after_sgd"]) < 1e-12
               for a, b in zip(stored["cases"], result["cases"]))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
