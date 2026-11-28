#!/usr/bin/env python3
"""
Apples-to-apples Compiled PyTorch baseline runner paired with AiDotNet.CrossFamilyCompiledSuite.

Implements the exact 12 Wave 1 + Wave 2 model families and tensor shapes using:
  - torch.compile(backend="inductor", mode="max-autotune")
  - Custom @triton.jit fused RMSNorm kernel when Triton is available on CUDA
  - Fused torch.nn.functional.cross_entropy / mse_loss
  - torch.optim.AdamW(..., fused=(device == "cuda"))
  - torch.nn.utils.clip_grad_norm_(..., max_norm=1.0, foreach=True)
  - Strict train/eval disjointness check (overlap == 0)
  - Raw, unmanipulated wall-clock timing via time.perf_counter_ns() with device synchronization
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Single CLI parser, used both for pre-import environment setup and at runtime.

    Accepts both ``--device cuda`` (the orchestrator's separate-token form) and ``--device=cuda``.
    """
    p = argparse.ArgumentParser(description="Paired Compiled PyTorch cross-family benchmark runner.")
    p.add_argument("--device", default="cpu", type=str.lower, choices=["cpu", "cuda", "gpu"])
    p.add_argument(
        "--mode",
        default="inductor-max-autotune",
        choices=["inductor-max-autotune", "inductor-default", "eager-control"],
    )
    p.add_argument("--precision", default="fp32", type=str.lower, choices=["fp32", "tf32", "bf16", "fp16"])
    p.add_argument("--wave", default="all", type=str.lower, choices=["all", "1", "2", "wave1", "wave2"])
    p.add_argument("--filter", default=None)
    p.add_argument("--smoke", action="store_true", dest="smoke")
    p.add_argument(
        "--cpu-dry-run",
        action="store_true",
        dest="cpu_dry_run",
        help="Smoke run forced onto the CPU (implies --smoke and --device cpu).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    if args.cpu_dry_run:
        args.smoke = True
        args.device = "cpu"
    if args.device == "gpu":
        args.device = "cuda"
    if args.steps < 1:
        p.error(f"--steps must be >= 1 (got {args.steps}); a run with no measured steps has no latency data.")
    if args.warmup < 0:
        p.error(f"--warmup must be >= 0 (got {args.warmup}).")
    if args.device == "cpu" and args.precision == "fp16":
        p.error("--precision fp16 is not supported with --device cpu: this runner enables FP16 autocast only on CUDA, "
                "so the workload would run in FP32 while being labelled FP16. Use fp32 or bf16 on CPU.")
    return args


def _require_approved_cuda_selection() -> str:
    """Require an explicit single-GPU selection that is not physical GPU 0.

    The process-local ``cuda:0`` ordinal says nothing about physical identity: with CUDA_VISIBLE_DEVICES unset every
    GPU is visible and ``cuda:0`` may be physical GPU 0, and without PCI_BUS_ID ordering the default
    FASTEST_FIRST enumeration can map index ``1`` to any card. So both must be set explicitly.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None or not vis.strip():
        raise RuntimeError(
            "Safety gate: --device cuda requires an explicit single-GPU selection. "
            "Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=1 (GPU 0 is off-limits)."
        )
    tokens = [t.strip() for t in vis.split(",") if t.strip()]
    if len(tokens) != 1:
        raise RuntimeError(f"Safety gate: CUDA_VISIBLE_DEVICES must select exactly one GPU (got '{vis}').")
    token = tokens[0]
    if not token.isdigit():
        raise RuntimeError(
            f"Safety gate: CUDA_VISIBLE_DEVICES must be a numeric PCI-bus-ordered index (got '{token}')."
        )
    if int(token) == 0:
        raise RuntimeError("Safety gate: GPU 0 is strictly off-limits. Set CUDA_VISIBLE_DEVICES=1.")
    if os.environ.get("CUDA_DEVICE_ORDER", "").strip().upper() != "PCI_BUS_ID":
        raise RuntimeError(
            "Safety gate: set CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA_VISIBLE_DEVICES indexes physical GPUs by bus order."
        )
    return token


def _preconfigure_environment(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        _require_approved_cuda_selection()
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")


_CLI_ARGS: Optional[argparse.Namespace] = parse_args(sys.argv[1:]) if __name__ == "__main__" else None
if _CLI_ARGS is not None:
    _preconfigure_environment(_CLI_ARGS)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.utils.parametrizations import weight_norm  # noqa: E402

HAS_TRITON = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_rmsnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_start = x_ptr + row_idx * stride_row
        x = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.math.rsqrt(mean_sq + eps)
        y = x * inv_rms * w
        tl.store(y_ptr + row_idx * stride_row + col_offsets, y, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, allow_triton_eval: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.allow_triton_eval = allow_triton_eval
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            HAS_TRITON
            and self.allow_triton_eval
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.is_contiguous()
            and self.dim <= 4096
        ):
            orig_shape = x.shape
            x2d = x.view(-1, self.dim)
            y2d = torch.empty_like(x2d)
            block = 1 << (self.dim - 1).bit_length()
            _triton_rmsnorm_fwd_kernel[(x2d.shape[0],)](
                x2d,
                self.weight,
                y2d,
                x2d.stride(0),
                self.dim,
                self.eps,
                BLOCK_SIZE=block,
            )
            return y2d.view(orig_shape)
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, allow_triton_eval=False)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm2 = RMSNorm(d_model, allow_triton_eval=False)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, s, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).contiguous().view(b, s, d))
        return x + self.fc2(F.silu(self.fc1(self.norm2(x))))


class CausalTransformerLM(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, heads: int, vocab: int, local_credit: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CausalTransformerBlock(d_model, heads) for _ in range(n_blocks)])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.local_credit = local_credit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.final_norm(x))

    def forward_local_credit_loss(self, x: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        total_loss: Optional[torch.Tensor] = None
        h = x
        for blk in self.blocks:
            h = blk(h.detach())
            local_logits = F.linear(
                self.final_norm(h),
                self.head.weight.detach(),
                self.head.bias.detach() if self.head.bias is not None else None,
            )
            blk_loss = F.cross_entropy(local_logits.reshape(-1, local_logits.size(-1)), target_idx.reshape(-1))
            total_loss = blk_loss if total_loss is None else (total_loss + blk_loss)
        final_logits = self.head(self.final_norm(h.detach()))
        head_loss = F.cross_entropy(final_logits.reshape(-1, final_logits.size(-1)), target_idx.reshape(-1))
        return head_loss if total_loss is None else (total_loss + head_loss)


class GatedLinearSsmBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkvg = nn.Linear(d_model, 4 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v, g = self.qkvg(x).chunk(4, dim=-1)
        gate = torch.sigmoid(g)
        q = F.elu(q.view(b, s, self.heads, self.head_dim)) + 1.0
        k = F.elu(k.view(b, s, self.heads, self.head_dim)) + 1.0
        v = (v * gate).view(b, s, self.heads, self.head_dim)
        kv = torch.einsum("bshd,bshm->bshdm", k, v).cumsum(dim=1)
        z = k.cumsum(dim=1)
        num = torch.einsum("bshd,bshdm->bshm", q, kv)
        den = (torch.einsum("bshd,bshd->bsh", q, z).unsqueeze(-1)).clamp_min(1e-5)
        mixed = (num / den).reshape(b, s, d)
        return self.norm(x + self.out_proj(mixed))


class GatedLinearSsmLM(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, heads: int, vocab: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([GatedLinearSsmBlock(d_model, heads) for _ in range(n_blocks)])
        self.norm = RMSNorm(d_model)
        self.fc_mid = nn.Linear(d_model, d_model * 2)
        self.head = nn.Linear(d_model * 2, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(F.silu(self.fc_mid(self.norm(x))))


class HouseholderLayer(nn.Module):
    def __init__(self, dim: int, k_refl: int, transpose: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.k_refl = k_refl
        self.transpose = transpose
        self.reflectors = nn.Parameter(torch.randn(k_refl, dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.reflectors / self.reflectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        order = range(self.k_refl - 1, -1, -1) if not self.transpose else range(self.k_refl)
        h = x
        for idx in order:
            ui = u[idx]
            proj = torch.matmul(h, ui).unsqueeze(-1)
            h = h - 2.0 * proj * ui
        return h


class HouseholderProjectionStack(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, k_refl: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for idx in range(n_blocks):
            layers.append(HouseholderLayer(d_model, k_refl, transpose=(idx % 2 == 1)))
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.GELU())
        layers.append(nn.Linear(d_model, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x) if self.down is not None else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class VisionResNetStageNet(nn.Module):
    def __init__(self, channels: int, spatial: int, width: int, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResNetBasicBlock(channels, width, stride=2),
            ResNetBasicBlock(width, width * 2, stride=2),
            nn.Flatten(),
            nn.Linear((width * 2) * (spatial // 4) * (spatial // 4), classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_mult: int = 4, use_silu: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * mlp_mult)
        self.act = nn.SiLU() if use_silu else nn.GELU()
        self.fc2 = nn.Linear(d_model * mlp_mult, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        x = x + self.proj(attn.transpose(1, 2).contiguous().view(b, s, d))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class VisionViTClassifier(nn.Module):
    def __init__(self, n_blocks: int, num_patches: int, d_model: int, heads: int, classes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads, mlp_mult=4, use_silu=False) for _ in range(n_blocks)])
        self.head = nn.Linear(num_patches * d_model, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.flatten(start_dim=1))


class TabularMlp(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_hidden: int, classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(cur, hidden))
            layers.append(nn.GELU())
            cur = hidden
        layers.append(nn.Linear(cur, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionDiTDenoiser(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads, mlp_mult=4, use_silu=True) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.out(self.norm(x))


class TimeSeriesWeightNormTcn(nn.Module):
    def __init__(self, channels: int, lookback: int, horizon: int) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels * 2, kernel_size=3, dilation=1, padding=1))
        self.conv2 = weight_norm(nn.Conv1d(channels * 2, channels * 2, kernel_size=3, dilation=2, padding=2))
        self.fc1 = nn.Linear((channels * 2) * lookback, channels * 4)
        self.fc2 = nn.Linear(channels * 4, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc2(F.gelu(self.fc1(h)))


class ConformerMacaronBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1_a = nn.Linear(d_model, d_model * 2)
        self.ff1_b = nn.Linear(d_model * 2, d_model)
        self.attn_block = EncoderBlock(d_model, heads, mlp_mult=2, use_silu=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1_b(F.silu(self.ff1_a(self.ff1_norm(x))))
        return self.attn_block(x)


class AudioConformerEncoder(nn.Module):
    def __init__(self, n_blocks: int, d_model: int, heads: int, vocab: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ConformerMacaronBlock(d_model, heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, adj: torch.Tensor, in_features: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.register_buffer("adj", adj)
        self.lin1 = nn.Linear(in_features, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(torch.matmul(self.adj, self.lin1(x)))
        h = F.relu(torch.matmul(self.adj, self.lin2(h)))
        return self.head(h)


def make_disjoint_features(shape: Tuple[int, ...], seed: int, partition_offset: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed((seed * 7919 + partition_offset * 104729) & 0x7FFFFFFF)
    t = (torch.rand(shape, generator=gen, dtype=torch.float32) * 2.0 - 1.0) * 0.5
    return t.to(device)


def make_class_targets(leading_dims: Tuple[int, ...], num_classes: int, seed: int, partition_offset: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed((seed * 6151 + partition_offset * 99991) & 0x7FFFFFFF)
    t = torch.randint(0, num_classes, leading_dims, generator=gen, dtype=torch.long)
    return t.to(device)


def verify_zero_row_overlap(train_x: torch.Tensor, eval_x: torch.Tensor) -> int:
    tx = train_x.detach().cpu().reshape(train_x.shape[0], -1).contiguous()
    ex = eval_x.detach().cpu().reshape(eval_x.shape[0], -1).contiguous()
    if tx.shape[1] != ex.shape[1]:
        return 0
    take = min(64, tx.shape[1])
    seen = set()
    for r in range(tx.shape[0]):
        raw_bytes = bytes(tx[r, :take].view(torch.uint8).tolist())
        seen.add(hashlib.blake2b(raw_bytes, digest_size=8).digest())
    overlap = 0
    for r in range(ex.shape[0]):
        raw_bytes = bytes(ex[r, :take].view(torch.uint8).tolist())
        if hashlib.blake2b(raw_bytes, digest_size=8).digest() in seen:
            overlap += 1
    if overlap != 0:
        raise RuntimeError(f"Adversarial Gate 7 violation: detected {overlap} overlapping train/eval windows.")
    return overlap


def percentile_sorted(sorted_vals: List[float], p: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = p * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize_latencies(samples_ms: List[float]) -> Dict[str, Any]:
    if not samples_ms:
        return {
            "samples_ms": [],
            "median_ms": 0.0,
            "iqr_ms": 0.0,
            "p95_ms": 0.0,
            "mean_ms": 0.0,
            "sd_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    s = sorted(samples_ms)
    n = len(s)
    med = s[n // 2] if (n % 2 == 1) else 0.5 * (s[n // 2 - 1] + s[n // 2])
    q25 = percentile_sorted(s, 0.25)
    q75 = percentile_sorted(s, 0.75)
    p95 = percentile_sorted(s, 0.95)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    return {
        "samples_ms": samples_ms,
        "median_ms": med,
        "iqr_ms": q75 - q25,
        "p95_ms": p95,
        "mean_ms": mean,
        "sd_ms": sd,
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_raw_action(device: torch.device, warmup: int, steps: int, fn: Callable[[], Any]) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
        sync_device(device)
    samples_ms: List[float] = []
    sync_device(device)
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        fn()
        sync_device(device)
        t1 = time.perf_counter_ns()
        samples_ms.append((t1 - t0) * 1e-6)
    return summarize_latencies(samples_ms)


def maybe_compile_module(module: nn.Module, mode_name: str, device: torch.device) -> Tuple[Any, str]:
    if mode_name == "eager-control":
        return module, "eager-control"
    compile_mode = "max-autotune" if mode_name == "inductor-max-autotune" else "default"
    try:
        compiled = torch.compile(module, backend="inductor", mode=compile_mode)
        return compiled, f"inductor:{compile_mode}"
    except Exception as exc:
        if device.type == "cuda":
            raise
        return module, f"cpu-fallback({type(exc).__name__})"


def build_autocast_ctx(device: torch.device, precision: str):
    p = precision.lower()
    if p == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if p == "fp16":
        if device.type != "cuda":
            raise ValueError("FP16 autocast is only enabled on CUDA; refusing to label an FP32 CPU run as FP16.")
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def configure_precision_flags(device: torch.device, precision: str) -> None:
    p = precision.lower()
    if p == "fp32":
        torch.set_float32_matmul_precision("highest")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def execute_standard_workload(
    args: argparse.Namespace,
    device: torch.device,
    workload_id: str,
    wave: int,
    spec: Dict[str, Any],
    model: nn.Module,
    b1_x: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    is_classification: bool,
    local_credit: bool = False,
    extra_notes: str = "",
    measure_b1: bool = True,
) -> Dict[str, Any]:
    overlap = verify_zero_row_overlap(train_x, eval_x)
    model = model.to(device)
    spec["parameter_count"] = int(sum(p.numel() for p in model.parameters()))

    use_fused_adamw = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=use_fused_adamw)
    params_list = [p for p in model.parameters() if p.requires_grad]

    def compute_loss(m: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if local_credit and hasattr(m, "forward_local_credit_loss"):
            return m.forward_local_credit_loss(x, y)
        out = m(x)
        if is_classification:
            return F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
        return F.mse_loss(out, y)

    model.eval()
    with torch.no_grad():
        initial_eval_loss = float(compute_loss(model, eval_x, eval_y).item())

    if not math.isfinite(initial_eval_loss):
        raise RuntimeError(
            f"Adversarial Gate 10 violation in '{workload_id}': non-finite initial eval loss ({initial_eval_loss})."
        )

    exec_model, compile_tag = maybe_compile_module(model, args.mode, device)
    infer_mode = compile_tag

    def run_infer(inp: torch.Tensor) -> None:
        with torch.no_grad():
            with build_autocast_ctx(device, args.precision):
                _ = exec_model(inp)

    try:
        run_infer(b1_x)
        sync_device(device)
    except Exception as exc:
        if device.type == "cuda":
            raise
        exec_model = model
        infer_mode = f"cpu-inductor-unavailable({type(exc).__name__})"

    b1_summary = (
        measure_raw_action(device, args.warmup, args.steps, lambda: run_infer(b1_x))
        if measure_b1
        else summarize_latencies([])
    )
    batch_infer_summary = measure_raw_action(device, args.warmup, args.steps, lambda: run_infer(eval_x))

    model.train()

    # The training entrypoint gets its own compilation: the local-credit objective does not go through
    # model.forward, so compiling the module would leave its training step eager while tagged compiled.
    if local_credit:
        def _local_credit_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return model.forward_local_credit_loss(x, y)  # type: ignore[operator]

        train_loss_fn, train_mode = maybe_compile_module(_local_credit_loss, args.mode, device)
    else:
        # The standard objective goes through model.forward, so it reuses the inference compilation.
        train_module, train_mode = exec_model, infer_mode

        def _standard_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = train_module(x)
            return (
                F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
                if is_classification
                else F.mse_loss(out, y)
            )

        train_loss_fn = _standard_loss

    def run_train_step(apply_update: bool = True) -> None:
        optimizer.zero_grad(set_to_none=True)
        with build_autocast_ctx(device, args.precision):
            loss = train_loss_fn(train_x, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_list, max_norm=1.0, foreach=True)
        if apply_update:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)

    # Probe the first training call before timing so a CPU compile failure falls back to eager. The probe is
    # counted inside the warmup budget so both runners commit exactly warmup + steps optimizer updates. With
    # --warmup 0 the probe runs forward + backward + clip but skips optimizer.step(), keeping the same failure
    # handling without committing an extra update.
    probe_commits_update = args.warmup > 0

    def _train_eager_fallback(exc: Exception) -> None:
        nonlocal train_loss_fn, train_mode
        if local_credit:
            def _eager_lc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return model.forward_local_credit_loss(x, y)  # type: ignore[operator]

            train_loss_fn = _eager_lc
        else:
            def _eager_std(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = model(x)
                return (
                    F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
                    if is_classification
                    else F.mse_loss(out, y)
                )

            train_loss_fn = _eager_std
        train_mode = f"cpu-train-fallback({type(exc).__name__})"

    try:
        run_train_step(apply_update=probe_commits_update)
        sync_device(device)
    except Exception as exc:
        if device.type == "cuda":
            raise
        # A failed probe never reached optimizer.step(), so no update was committed; clear partial grads and
        # rerun the probe eagerly so the update budget stays exact.
        optimizer.zero_grad(set_to_none=True)
        _train_eager_fallback(exc)
        run_train_step(apply_update=probe_commits_update)
        sync_device(device)

    remaining_warmup = args.warmup - 1 if probe_commits_update else 0
    train_summary = measure_raw_action(device, remaining_warmup, args.steps, run_train_step)

    model.eval()
    with torch.no_grad():
        final_eval_loss = float(compute_loss(model, eval_x, eval_y).item())
    if not math.isfinite(final_eval_loss):
        raise RuntimeError(
            f"Adversarial Gate 10 violation in '{workload_id}': non-finite final eval loss ({final_eval_loss})."
        )

    compile_tag = infer_mode if infer_mode == train_mode else f"infer={infer_mode};train={train_mode}"

    batch_size = int(train_x.shape[0])
    notes_parts = [
        f"compile={compile_tag}",
        f"triton={HAS_TRITON}",
        f"fused_adamw={use_fused_adamw}",
        "foreach_clip=True",
    ]
    if extra_notes:
        notes_parts.append(extra_notes)

    return {
        "workload_id": workload_id,
        "wave": wave,
        "framework": "PyTorch",
        "execution_mode": compile_tag,
        "requested_mode": args.mode,
        "inference_execution_mode": infer_mode,
        "train_execution_mode": train_mode,
        "precision_tier": args.precision.upper(),
        "device": args.device,
        "seed": args.seed,
        "batch_size": batch_size,
        "train_eval_window_overlap": overlap,
        "model_spec": spec,
        "b1_available": measure_b1,
        "b1_inference_latency": b1_summary,
        "batched_inference_latency": batch_infer_summary,
        "batched_train_step_latency": train_summary,
        "batched_infer_items_per_sec": (
            batch_size * 1000.0 / batch_infer_summary["median_ms"]
            if batch_infer_summary["median_ms"] > 0
            else 0.0
        ),
        "batched_train_items_per_sec": (
            batch_size * 1000.0 / train_summary["median_ms"]
            if train_summary["median_ms"] > 0
            else 0.0
        ),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "managed_alloc_bytes_per_step": 0.0,
        "notes": "; ".join(notes_parts),
    }


def run_optimizer_sweep(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    batch = 8 if args.smoke else 32
    in_dim = 32 if args.smoke else 256
    hid_dim = 64 if args.smoke else 512
    classes = 8 if args.smoke else 32

    train_x = make_disjoint_features((batch, in_dim), args.seed, 41, device)
    train_y = make_class_targets((batch,), classes, args.seed, 42, device)
    eval_x = make_disjoint_features((batch, in_dim), args.seed, 43, device)
    eval_y = make_class_targets((batch,), classes, args.seed, 44, device)
    b1_x = make_disjoint_features((1, in_dim), args.seed, 45, device)
    overlap = verify_zero_row_overlap(train_x, eval_x)

    def make_mlp() -> TabularMlp:
        # Same seed -> same initial weights for every optimizer case.
        torch.manual_seed(args.seed)
        return TabularMlp(in_dim, hid_dim, n_hidden=2, classes=classes).to(device)

    def eval_loss_of(m: nn.Module) -> float:
        m.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(m(eval_x), eval_y).item())
        m.train()
        return value

    # Exactly the optimizer names the AiDotNet paired sweep ("wave1_optimizer_base_sweep") times, with the same
    # timed work in both arms: forward + loss + backward + optimizer step, no gradient clipping. Framework-specific
    # optimizers (ASGD, Adafactor here; Adam8Bit, Lion, LAMB, ... in AiDotNet) are not in the paired median.
    use_cuda_fused = device.type == "cuda"
    opt_builders: List[Tuple[str, Callable[[Any], torch.optim.Optimizer]]] = [
        ("AdamW", lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=use_cuda_fused)),
        ("Adam", lambda ps: torch.optim.Adam(ps, lr=1e-3, fused=use_cuda_fused)),
        ("SGD", lambda ps: torch.optim.SGD(ps, lr=1e-2, foreach=True)),
        ("Momentum", lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9, foreach=True)),
        ("NAG", lambda ps: torch.optim.SGD(ps, lr=1e-2, momentum=0.9, nesterov=True, foreach=True)),
        ("RMSProp", lambda ps: torch.optim.RMSprop(ps, lr=1e-3, foreach=True)),
        ("Adagrad", lambda ps: torch.optim.Adagrad(ps, lr=1e-2, foreach=True)),
        ("AdaDelta", lambda ps: torch.optim.Adadelta(ps, lr=1.0, foreach=True)),
        ("AdaMax", lambda ps: torch.optim.Adamax(ps, lr=2e-3, foreach=True)),
        ("AMSGrad", lambda ps: torch.optim.Adam(ps, lr=1e-3, amsgrad=True, fused=use_cuda_fused)),
        ("Nadam", lambda ps: torch.optim.NAdam(ps, lr=1e-3, foreach=True)),
        ("RAdam", lambda ps: torch.optim.RAdam(ps, lr=1e-3, foreach=True)),
        ("Rprop", lambda ps: torch.optim.Rprop(ps, lr=1e-2, foreach=True)),
    ]

    per_opt_medians: List[float] = []
    per_opt_initial_loss: List[float] = []
    per_opt_final_loss: List[float] = []
    per_opt_notes: List[str] = []
    param_count = 0
    iters = 2 if args.smoke else 5

    for opt_name, build_opt in opt_builders:
        net = make_mlp()
        param_count = sum(p.numel() for p in net.parameters())
        params = [p for p in net.parameters() if p.requires_grad]
        opt = build_opt(params)
        initial_loss = eval_loss_of(net)

        def step_fn() -> None:
            opt.zero_grad(set_to_none=True)
            out = net(train_x)
            loss = F.cross_entropy(out, train_y)
            loss.backward()
            opt.step()

        summary = measure_raw_action(device, warmup=1, steps=iters, fn=step_fn)
        for p in params:
            if not torch.isfinite(p).all():
                raise RuntimeError(f"PyTorch optimizer '{opt_name}' produced non-finite parameters.")
        final_loss = eval_loss_of(net)
        if not (math.isfinite(initial_loss) and math.isfinite(final_loss)):
            raise RuntimeError(
                f"Adversarial Gate 10 violation in 'wave1_optimizer_base_sweep' optimizer '{opt_name}': "
                f"non-finite eval loss ({initial_loss} -> {final_loss})."
            )
        per_opt_medians.append(summary["median_ms"])
        per_opt_initial_loss.append(initial_loss)
        per_opt_final_loss.append(final_loss)
        per_opt_notes.append(
            f"{opt_name}:step_median_ms={summary['median_ms']:.4f},loss={initial_loss:.4f}->{final_loss:.4f}"
        )

    # Inference latency is optimizer-independent: time a freshly built (seeded) copy of the same MLP, with
    # autograd disabled exactly as the standard inference path does.
    probe = make_mlp()
    probe.eval()

    def probe_infer(inp: torch.Tensor) -> None:
        with torch.no_grad():
            _ = probe(inp)

    b1_summary = measure_raw_action(device, 1, iters, lambda: probe_infer(b1_x))
    batch_infer_summary = measure_raw_action(device, 1, iters, lambda: probe_infer(eval_x))
    sweep_summary = summarize_latencies(per_opt_medians)

    return {
        "workload_id": "wave1_optimizer_base_sweep",
        "wave": 1,
        "framework": "PyTorch",
        "execution_mode": f"foreach/fused({len(opt_builders)}opts)",
        "requested_mode": args.mode,
        "inference_execution_mode": "eager",
        "train_execution_mode": "eager",
        "precision_tier": args.precision.upper(),
        "device": args.device,
        "seed": args.seed,
        "batch_size": batch,
        "train_eval_window_overlap": overlap,
        "model_spec": {
            "architecture": f"PyTorchForeachFusedOptimizer-PairedSweep-{len(opt_builders)}Opts",
            "init": "SeededRandom",
            "layers": 3,
            "hidden_dim": hid_dim,
            "vocab_or_classes": classes,
            "context_or_spatial": in_dim,
            "parameter_count": param_count,
        },
        "b1_inference_latency": b1_summary,
        "batched_inference_latency": batch_infer_summary,
        "batched_train_step_latency": sweep_summary,
        "batched_infer_items_per_sec": (
            batch * 1000.0 / batch_infer_summary["median_ms"] if batch_infer_summary["median_ms"] > 0 else 0.0
        ),
        "batched_train_items_per_sec": (
            batch * 1000.0 / sweep_summary["median_ms"] if sweep_summary["median_ms"] > 0 else 0.0
        ),
        # Median across optimizer cases of each case's own pre-/post-run eval loss.
        "initial_eval_loss": summarize_latencies(per_opt_initial_loss)["median_ms"],
        "final_eval_loss": summarize_latencies(per_opt_final_loss)["median_ms"],
        "managed_alloc_bytes_per_step": 0.0,
        "notes": (
            f"Paired sweep of {len(opt_builders)} PyTorch fused/foreach optimizers; "
            "timed op = forward+loss+backward+step (no clip). " + "; ".join(per_opt_notes)
        ),
    }


def build_all_workloads(args: argparse.Namespace, device: torch.device) -> List[Tuple[int, str, Callable[[], Dict[str, Any]]]]:
    s = args.smoke
    seed = args.seed
    workloads: List[Tuple[int, str, Callable[[], Dict[str, Any]]]] = []

    for local_credit, wid in [
        (False, "wave1_causal_transformer_lm_backprop"),
        (True, "wave1_causal_transformer_lm_local_credit"),
    ]:
        def _run_ct(lc: bool = local_credit, w_id: str = wid) -> Dict[str, Any]:
            batch, seq, d_model, heads, n_blocks, vocab = (
                (2, 16, 64, 4, 2, 128) if s else (8, 64, 256, 8, 4, 1024)
            )
            m = CausalTransformerLM(n_blocks, d_model, heads, vocab, local_credit=lc)
            return execute_standard_workload(
                args,
                device,
                w_id,
                1,
                {
                    "architecture": "CausalTransformerLM-TiedReadoutLocalCredit" if lc else "CausalTransformerLM-Backprop",
                    "init": "SeededRandom",
                    "layers": n_blocks,
                    "hidden_dim": d_model,
                    "vocab_or_classes": vocab,
                    "context_or_spatial": seq,
                },
                m,
                make_disjoint_features((1, seq, d_model), seed, 10, device),
                make_disjoint_features((batch, seq, d_model), seed, 11, device),
                make_class_targets((batch, seq), vocab, seed, 12, device),
                make_disjoint_features((batch, seq, d_model), seed, 13, device),
                make_class_targets((batch, seq), vocab, seed, 14, device),
                is_classification=True,
                local_credit=lc,
            )

        workloads.append((1, wid, _run_ct))

    def _run_gla() -> Dict[str, Any]:
        batch, seq, d_model, heads, n_blocks, vocab = (
            (2, 16, 64, 4, 2, 128) if s else (8, 64, 256, 8, 4, 1024)
        )
        m = GatedLinearSsmLM(n_blocks, d_model, heads, vocab)
        return execute_standard_workload(
            args,
            device,
            "wave1_gated_linear_ssm_lm",
            1,
            {
                "architecture": "GatedLinearAttentionSSM-LM",
                "init": "SeededRandom",
                "layers": n_blocks,
                "hidden_dim": d_model,
                "vocab_or_classes": vocab,
                "context_or_spatial": seq,
            },
            m,
            make_disjoint_features((1, seq, d_model), seed, 20, device),
            make_disjoint_features((batch, seq, d_model), seed, 21, device),
            make_class_targets((batch, seq), vocab, seed, 22, device),
            make_disjoint_features((batch, seq, d_model), seed, 23, device),
            make_class_targets((batch, seq), vocab, seed, 24, device),
            is_classification=True,
        )

    workloads.append((1, "wave1_gated_linear_ssm_lm", _run_gla))

    def _run_hh() -> Dict[str, Any]:
        batch, seq, d_model, k_refl, n_blocks = (4, 16, 64, 4, 4) if s else (16, 64, 256, 4, 8)
        m = HouseholderProjectionStack(n_blocks, d_model, k_refl)
        return execute_standard_workload(
            args,
            device,
            "wave1_householder_orthogonal_projection",
            1,
            {
                "architecture": f"HouseholderOrthogonalStack-k{k_refl}",
                "init": "SeededRandom",
                "layers": n_blocks,
                "hidden_dim": d_model,
                "vocab_or_classes": d_model,
                "context_or_spatial": seq,
            },
            m,
            make_disjoint_features((1, seq, d_model), seed, 30, device),
            make_disjoint_features((batch, seq, d_model), seed, 31, device),
            make_disjoint_features((batch, seq, d_model), seed, 32, device),
            make_disjoint_features((batch, seq, d_model), seed, 33, device),
            make_disjoint_features((batch, seq, d_model), seed, 34, device),
            is_classification=False,
        )

    workloads.append((1, "wave1_householder_orthogonal_projection", _run_hh))
    workloads.append((1, "wave1_optimizer_base_sweep", lambda: run_optimizer_sweep(args, device)))

    def _run_resnet() -> Dict[str, Any]:
        batch, channels, spatial, width, classes = (2, 3, 16, 16, 10) if s else (16, 3, 32, 64, 100)
        m = VisionResNetStageNet(channels, spatial, width, classes)
        return execute_standard_workload(
            args,
            device,
            "wave1_vision_resnet",
            1,
            {
                "architecture": "VisionConvBatchNormStageNet",
                "init": "SeededRandom",
                "layers": 8,
                "hidden_dim": width * 2,
                "vocab_or_classes": classes,
                "context_or_spatial": spatial,
            },
            m,
            make_disjoint_features((1, channels, spatial, spatial), seed, 50, device),
            make_disjoint_features((batch, channels, spatial, spatial), seed, 51, device),
            make_class_targets((batch,), classes, seed, 52, device),
            make_disjoint_features((batch, channels, spatial, spatial), seed, 53, device),
            make_class_targets((batch,), classes, seed, 54, device),
            is_classification=True,
        )

    workloads.append((1, "wave1_vision_resnet", _run_resnet))

    def _run_vit() -> Dict[str, Any]:
        batch, num_patches, d_model, heads, n_blocks, classes = (
            (2, 16, 64, 4, 2, 10) if s else (16, 64, 192, 6, 4, 100)
        )
        m = VisionViTClassifier(n_blocks, num_patches, d_model, heads, classes)
        return execute_standard_workload(
            args,
            device,
            "wave1_vision_vit",
            1,
            {
                "architecture": "VisionTransformerEncoder",
                "init": "SeededRandom",
                "layers": n_blocks,
                "hidden_dim": d_model,
                "vocab_or_classes": classes,
                "context_or_spatial": num_patches,
            },
            m,
            make_disjoint_features((1, num_patches, d_model), seed, 60, device),
            make_disjoint_features((batch, num_patches, d_model), seed, 61, device),
            make_class_targets((batch,), classes, seed, 62, device),
            make_disjoint_features((batch, num_patches, d_model), seed, 63, device),
            make_class_targets((batch,), classes, seed, 64, device),
            is_classification=True,
        )

    workloads.append((1, "wave1_vision_vit", _run_vit))

    def _run_mlp() -> Dict[str, Any]:
        batch, in_features, hidden, n_hidden, classes = (8, 64, 128, 4, 8) if s else (64, 256, 512, 4, 32)
        m = TabularMlp(in_features, hidden, n_hidden, classes)
        return execute_standard_workload(
            args,
            device,
            "wave1_tabular_mlp",
            1,
            {
                "architecture": "TabularMLP-GELU",
                "init": "SeededRandom",
                "layers": n_hidden + 1,
                "hidden_dim": hidden,
                "vocab_or_classes": classes,
                "context_or_spatial": in_features,
            },
            m,
            make_disjoint_features((1, in_features), seed, 70, device),
            make_disjoint_features((batch, in_features), seed, 71, device),
            make_class_targets((batch,), classes, seed, 72, device),
            make_disjoint_features((batch, in_features), seed, 73, device),
            make_class_targets((batch,), classes, seed, 74, device),
            is_classification=True,
        )

    workloads.append((1, "wave1_tabular_mlp", _run_mlp))

    def _run_dit() -> Dict[str, Any]:
        batch, latent_tokens, d_model, heads, n_blocks = (2, 16, 64, 4, 2) if s else (8, 64, 256, 8, 4)
        m = DiffusionDiTDenoiser(n_blocks, d_model, heads)
        return execute_standard_workload(
            args,
            device,
            "wave2_diffusion_dit_unet",
            2,
            {
                "architecture": "DiffusionDiTDenoiserBlocks",
                "init": "SeededRandom",
                "layers": n_blocks,
                "hidden_dim": d_model,
                "vocab_or_classes": d_model,
                "context_or_spatial": latent_tokens,
            },
            m,
            make_disjoint_features((1, latent_tokens, d_model), seed, 80, device),
            make_disjoint_features((batch, latent_tokens, d_model), seed, 81, device),
            make_disjoint_features((batch, latent_tokens, d_model), seed, 82, device),
            make_disjoint_features((batch, latent_tokens, d_model), seed, 83, device),
            make_disjoint_features((batch, latent_tokens, d_model), seed, 84, device),
            is_classification=False,
        )

    workloads.append((2, "wave2_diffusion_dit_unet", _run_dit))

    def _run_tcn() -> Dict[str, Any]:
        batch, channels, lookback, horizon = (2, 8, 24, 12) if s else (16, 32, 96, 24)
        m = TimeSeriesWeightNormTcn(channels, lookback, horizon)
        return execute_standard_workload(
            args,
            device,
            "wave2_timeseries_patchtst_tcn",
            2,
            {
                "architecture": "TimeSeriesDilatedWeightNormTCN",
                "init": "SeededRandom",
                "layers": 5,
                "hidden_dim": channels * 2,
                "vocab_or_classes": horizon,
                "context_or_spatial": lookback,
            },
            m,
            make_disjoint_features((1, channels, lookback), seed, 90, device),
            make_disjoint_features((batch, channels, lookback), seed, 91, device),
            make_disjoint_features((batch, horizon), seed, 92, device),
            make_disjoint_features((batch, channels, lookback), seed, 93, device),
            make_disjoint_features((batch, horizon), seed, 94, device),
            is_classification=False,
        )

    workloads.append((2, "wave2_timeseries_patchtst_tcn", _run_tcn))

    def _run_conformer() -> Dict[str, Any]:
        batch, frames, d_model, heads, n_blocks, vocab = (
            (2, 20, 64, 4, 2, 32) if s else (8, 80, 192, 6, 4, 128)
        )
        m = AudioConformerEncoder(n_blocks, d_model, heads, vocab)
        return execute_standard_workload(
            args,
            device,
            "wave2_audio_conformer",
            2,
            {
                "architecture": "AudioConformerMacaronEncoder",
                "init": "SeededRandom",
                "layers": n_blocks,
                "hidden_dim": d_model,
                "vocab_or_classes": vocab,
                "context_or_spatial": frames,
            },
            m,
            make_disjoint_features((1, frames, d_model), seed, 100, device),
            make_disjoint_features((batch, frames, d_model), seed, 101, device),
            make_class_targets((batch, frames), vocab, seed, 102, device),
            make_disjoint_features((batch, frames, d_model), seed, 103, device),
            make_class_targets((batch, frames), vocab, seed, 104, device),
            is_classification=True,
        )

    workloads.append((2, "wave2_audio_conformer", _run_conformer))

    def _run_gnn() -> Dict[str, Any]:
        batch_graphs, num_nodes, node_features, hidden, classes = (
            (2, 16, 32, 64, 4) if s else (8, 64, 128, 128, 16)
        )
        total_nodes = batch_graphs * num_nodes
        adj = torch.zeros((total_nodes, total_nodes), dtype=torch.float32, device=device)
        for i in range(total_nodes):
            adj[i, i] = 0.25
            adj[i, (i - 1) % total_nodes] = 0.25
            adj[i, (i + 1) % total_nodes] = 0.25
            adj[i, (i + 4) % total_nodes] = 0.25
        m = GraphConvolutionalNetwork(adj, node_features, hidden, classes)
        return execute_standard_workload(
            args,
            device,
            "wave2_gnn_gcn_gat",
            2,
            {
                "architecture": "GraphConvolutionalNetwork-2Layer",
                "init": "SeededRandom",
                "layers": 3,
                "hidden_dim": hidden,
                "vocab_or_classes": classes,
                "context_or_spatial": total_nodes,
            },
            m,
            make_disjoint_features((total_nodes, node_features), seed, 110, device),
            make_disjoint_features((total_nodes, node_features), seed, 111, device),
            make_class_targets((total_nodes,), classes, seed, 112, device),
            make_disjoint_features((total_nodes, node_features), seed, 113, device),
            make_class_targets((total_nodes,), classes, seed, 114, device),
            is_classification=True,
            extra_notes=(
                f"FullGraphNodes={total_nodes}; B=1 unavailable "
                f"(transductive full-graph GCN: every inference covers all {total_nodes} nodes)"
            ),
            measure_b1=False,
        )

    workloads.append((2, "wave2_gnn_gcn_gat", _run_gnn))
    return workloads


def main(args: argparse.Namespace) -> int:
    if args.smoke:
        args.warmup = min(args.warmup, 2)
        args.steps = min(args.steps, 3)

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        _require_approved_cuda_selection()
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Safety gate: expected exactly one visible CUDA device, found {torch.cuda.device_count()}."
            )
        bound_name = torch.cuda.get_device_name(0)
        if "3080" in bound_name:
            raise RuntimeError(f"Safety gate: bound to '{bound_name}' (GPU 0 RTX 3080 is off-limits).")
        print(f"[PyTorch.CrossFamilyCompiledSuite] Bound CUDA device: {bound_name}")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    configure_precision_flags(device, args.precision)
    print(
        f"[PyTorch.CrossFamilyCompiledSuite] torch={torch.__version__} device={device} "
        f"mode={args.mode} precision={args.precision} wave={args.wave} smoke={args.smoke} seed={args.seed} triton={HAS_TRITON}"
    )

    run_w1 = args.wave in ("all", "1", "wave1")
    run_w2 = args.wave in ("all", "2", "wave2")
    records: List[Dict[str, Any]] = []

    for wave, wid, fn in build_all_workloads(args, device):
        if wave == 1 and not run_w1:
            continue
        if wave == 2 and not run_w2:
            continue
        if args.filter and args.filter.lower() not in wid.lower():
            continue
        rec = fn()
        records.append(rec)
        print(
            f"  {rec['workload_id']:<42} | params={rec['model_spec']['parameter_count']:>9} | "
            + (
                f"B=1 med={rec['b1_inference_latency']['median_ms']:>8.3f} ms | "
                if rec.get("b1_available", True)
                else f"B=1 med={'n/a':>8}    | "
            )
            + f"BatchInfer med={rec['batched_inference_latency']['median_ms']:>8.3f} ms "
            f"({rec['batched_infer_items_per_sec']:>8.1f} it/s) | "
            f"TrainStep med={rec['batched_train_step_latency']['median_ms']:>8.3f} ms | "
            f"loss {rec['initial_eval_loss']:.4f}->{rec['final_eval_loss']:.4f}"
        )

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[PyTorch.CrossFamilyCompiledSuite] Wrote {len(records)} records -> {out_path}")

    return 0


if __name__ == "__main__":
    assert _CLI_ARGS is not None
    raise SystemExit(main(_CLI_ARGS))
