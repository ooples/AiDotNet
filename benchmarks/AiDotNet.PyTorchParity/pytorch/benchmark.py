"""PyTorch side of the AiDotNet ⇄ PyTorch parity benchmark.

Twin of the C# harness (../Program.cs): same four reference models
(MLP / CNN / LSTM / Transformer) with matching layer shapes, the same
training + multi-batch inference measurement loop, and the same JSON schema,
so the two reports compare directly via compare.py.

The performance target uses --compile (TorchInductor), including first-epoch
and steady-state measurements. Without that flag this script measures an eager
baseline. Pin both sides to the same device and CPU thread count:
    python benchmark.py --threads 8 --compile ...
    dotnet run ... (set AIDOTNET_CPU_THREADS=8)

Usage:
    pip install -r requirements.txt
    python benchmark.py --models mlp,cnn,lstm,transformer --device cpu --compile \
        --threads 8 --output ../results/pytorch.json
"""

from __future__ import annotations

import contextlib
import argparse
import json
import math
import re
import os
import platform
import random
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_telemetry import mark_phase

BATCH_SIZES = [1, 8, 32, 128]


@dataclass
class ResourceSummary:
    cpu_percent_avg: float
    rss_mb_peak: float
    gpu_util_percent_avg: float | None
    gpu_mem_mb_peak: float | None
    # This process's caching-allocator peaks over the phase (torch.cuda.max_memory_allocated / max_memory_reserved):
    # unlike gpu_mem_mb_peak (device-wide nvidia-smi), they exclude the CUDA context and every other process.
    gpu_allocated_mb_peak: float | None = None
    gpu_reserved_mb_peak: float | None = None


@dataclass
class TrainingResult:
    epoch_seconds: list[float]
    total_seconds: float
    steady_state_epoch_seconds_avg: float
    steady_state_epoch_seconds_median: float
    steady_state_epoch_seconds_p25: float
    steady_state_epoch_seconds_p75: float
    steps_per_epoch: int
    gradient_seconds_avg: float
    data_loading_seconds_avg: float
    resources: ResourceSummary
    # Loss after the first and the last timed step: proof the step trains, read off the device only after timing.
    # A non-finite loss is the string "NaN", "Infinity" or "-Infinity", as in the C# report: bare NaN tokens are not JSON.
    first_loss: float | str | None = None
    last_loss: float | str | None = None


@dataclass
class InferenceBatchResult:
    batch_size: int
    warmup_seconds_avg: float
    steady_state_latency_ms_avg: float
    steady_state_latency_ms_p95: float
    steady_state_latency_ms_median: float
    steady_state_latency_ms_p25: float
    steady_state_latency_ms_p75: float
    throughput_samples_per_second: float
    memory_mb_peak: float
    gpu_tensor_allocated_mb_peak: float | None = None


@dataclass
class ModelResult:
    model: str
    device: str
    parameters: int
    training: TrainingResult
    inference: list[InferenceBatchResult]
    # Set when the two rows compute different functions despite matching sizes, so the timing is not a parity result.
    note: str = ""


class ResourceMonitor:
    def __init__(self, sample_seconds: float = 0.1, track_gpu: bool = True, gpu_id: str | None = None,
                 torch_device: torch.device | None = None) -> None:
        self.sample_seconds = sample_seconds
        # The CUDA device whose allocator peaks the phase reports (None: not a CUDA run).
        self.torch_device = torch_device if torch_device is not None and torch_device.type == "cuda" else None
        # The GPU the benchmark runs on, as an nvidia-smi --id (its UUID): unqualified, nvidia-smi lists every GPU and
        # the first line was read, which on a two-GPU machine is not necessarily this one (the C# side does the same).
        self.gpu_id = gpu_id
        # nvidia-smi reports SYSTEM-WIDE GPU utilization/memory, not this
        # process's. On a CPU-only benchmark run that attributes whatever the
        # desktop GPU happens to be doing (browser, compositor, other apps)
        # to the benchmark — a committed CPU artifact once carried
        # gpu_util=5% / gpu_mem=792MB despite device=cpu. Only sample the GPU
        # when the benchmark actually targets it.
        self.track_gpu = track_gpu
        self.process = psutil.Process(os.getpid())
        self.cpu: list[float] = []
        self.rss_mb: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> ResourceMonitor:
        if self.torch_device is not None:
            torch.cuda.synchronize(self.torch_device)
            torch.cuda.reset_peak_memory_stats(self.torch_device)
        self.process.cpu_percent(interval=None)
        if os.environ.get("AIDOTNET_BENCH_EXTERNAL_MONITOR") != "1":
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.cpu.append(self.process.cpu_percent(interval=None))
            self.rss_mb.append(self.process.memory_info().rss / 1024 / 1024)
            if self.track_gpu:
                gpu = query_nvidia_smi(self.gpu_id)
                if gpu is not None:
                    util, mem = gpu
                    self.gpu_util.append(util)
                    self.gpu_mem.append(mem)
            time.sleep(self.sample_seconds)

    def summary(self) -> ResourceSummary:
        return ResourceSummary(
            cpu_percent_avg=round(statistics.fmean(self.cpu), 3) if self.cpu else 0.0,
            rss_mb_peak=round(max(self.rss_mb), 3) if self.rss_mb else 0.0,
            gpu_util_percent_avg=round(statistics.fmean(self.gpu_util), 3) if self.gpu_util else None,
            gpu_mem_mb_peak=round(max(self.gpu_mem), 3) if self.gpu_mem else None,
            gpu_allocated_mb_peak=round(torch.cuda.max_memory_allocated(self.torch_device) / 1024 / 1024, 3)
            if self.torch_device is not None else None,
            gpu_reserved_mb_peak=round(torch.cuda.max_memory_reserved(self.torch_device) / 1024 / 1024, 3)
            if self.torch_device is not None else None,
        )


def cuda_smi_id(device: torch.device) -> str | None:
    """nvidia-smi's --id for a CUDA device: its UUID, as torch reports it (None when torch cannot tell)."""
    if device.type != "cuda":
        return None
    try:
        return "GPU-" + str(torch.cuda.get_device_properties(device).uuid)
    except (RuntimeError, AttributeError):
        return None


def query_nvidia_smi(gpu_id: str | None = None) -> tuple[float, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ] + ([f"--id={gpu_id}"] if gpu_id else []),
            text=True,
            timeout=1,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first = output.strip().splitlines()[0]
    util, mem = [float(part.strip()) for part in first.split(",")]
    return util, mem


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.shape[0], -1))


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class LSTMClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.head = nn.Linear(64, 10)
        # AiDotNet's LSTMLayer has ONE bias per gate; PyTorch's two (bias_ih + bias_hh) are the same function but
        # each takes its own Adam step, so the effective bias moves twice as far per step (measured: every bias
        # update off by exactly 50%). Zeroing and freezing bias_hh makes the parameterization identical.
        self.lstm.bias_hh_l0.data.zero_()
        self.lstm.bias_hh_l0.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Same block as AiDotNet's TransformerEncoderBlock: Pre-LN (norm_first), no dropout (the C# side runs
        # dropoutRate 0; PyTorch's default 0.1 was extra work and a different function), and no Q/K/V bias --
        # AiDotNet's MultiHeadAttentionLayer has only an output bias, so in_proj_bias is zeroed and frozen.
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.proj = nn.Linear(32, 64)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)
        for block in self.encoder.layers:
            block.self_attn.in_proj_bias.data.zero_()
            block.self_attn.in_proj_bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.proj(x))
        return self.head(encoded.mean(dim=1))


# Both apply the reset gate to the recurrent product (n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), AiDotNet's GRULayer
# default resetAfter: true). nn.GRU also carries a recurrent bias b_hh; AiDotNet's GRULayer has one bias per gate.
GRU_NOTE = ("Same reset-after recurrence and weight shapes; nn.GRU has a second (recurrent) bias per gate that "
            "AiDotNet's GRULayer folds into one, so a timing comparison rather than bitwise parity.")
MODEL_NOTES = {
    "gru": GRU_NOTE,
    "gat": "One sample is the whole Cora-sized graph (2,708 nodes); run at --batch-size 1 --inference-batches 1. "
           "Dense masked attention on both sides.",
    "dcgan": "One training step updates the discriminator then the generator; the loss is the discriminator's.",
    "wav2vec2-base": "Feature encoder frozen and LayerDrop 0 on both sides; CTC loss averaged per utterance (not "
                     "transformers' sum or target-length reduction).",
    "hifigan-v1": "One utterance per step: run at --batch-size 1. A step updates the discriminators then the "
                  "generator; the loss is the generator's (adversarial + 2 x feature matching + 45 x mel L1).",
}


class GRUClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TinyViT(nn.Module):
    """ViT at 32x32, patch 4 (64 tokens + CLS), d=192, 6 pre-LN layers, 3 heads, MLP 768: the C# row's
    VisionTransformerOptions. Classification reads the CLS token, as AiDotNet's VisionTransformer does."""

    def __init__(self, image: int = 32, patch: int = 4, dim: int = 192, depth: int = 6, heads: int = 3,
                 mlp: int = 768, classes: int = 10) -> None:
        super().__init__()
        tokens = (image // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1) + self.pos
        return self.head(self.norm(self.encoder(tokens))[:, 0])


LM_VOCAB, LM_POSITIONS, LM_WIDTH = 512, 64, 128


class PreNormBlock(nn.Module):
    """AiDotNet's PreLNTransformerBlock: x + MHA(RMSNorm(x)), then h + Linear(GELU(Linear(RMSNorm(h)))). RMSNorm eps
    1e-6 with a gain only; AiDotNet's attention has no Q/K/V bias, so in_proj_bias is zeroed and frozen."""

    def __init__(self, width: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-6)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.attn.in_proj_bias.data.zero_()
        self.attn.in_proj_bias.requires_grad_(False)
        self.norm2 = nn.RMSNorm(width, eps=1e-6)
        self.up = nn.Linear(width, ffn)
        self.down = nn.Linear(ffn, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
        x = x + self.attn(n, n, n, attn_mask=mask, need_weights=False)[0]
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x))))


class DecoderLM(nn.Module):
    """The C# "decoder-lm" row: Embedding(512, 128) + 2 causal pre-norm blocks + RMSNorm + Linear(128, 512)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(LM_VOCAB, LM_WIDTH)
        self.blocks = nn.Sequential(*[PreNormBlock(LM_WIDTH, 4, 4 * LM_WIDTH) for _ in range(2)])
        self.norm = nn.RMSNorm(LM_WIDTH, eps=1e-6)
        self.head = nn.Linear(LM_WIDTH, LM_VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(self.embed(x))))


def lm_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequences following t[i+1] = (7 t[i] + 3) mod V from a random first token; the target is the next token at
    every position (the C# row's LoadSyntheticBatch)."""
    tokens = torch.empty(batch_size, LM_POSITIONS + 1, dtype=torch.long)
    tokens[:, 0] = torch.randint(0, LM_VOCAB, (batch_size,))
    for t in range(LM_POSITIONS):
        tokens[:, t + 1] = (7 * tokens[:, t] + 3) % LM_VOCAB
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def kipf_graph(nodes: int = 32) -> torch.Tensor:
    """A ring over the nodes plus a chord i -> i+5 from every third node (so degrees differ), with self-loops,
    normalized D^-1/2 (A + I) D^-1/2 -- the same graph as the C# row's KipfGraph."""
    a = torch.zeros(nodes, nodes, dtype=torch.float64)
    for i in range(nodes):
        a[i, i] = 1
        a[i, (i + 1) % nodes] = a[(i + 1) % nodes, i] = 1
        if i % 3 == 0:
            a[i, (i + 5) % nodes] = a[(i + 5) % nodes, i] = 1
    inv_sqrt = a.sum(dim=1).rsqrt()
    return (inv_sqrt[:, None] * a * inv_sqrt[None, :]).float()


GCN_NODES, GCN_FEATURES = 32, 16


def gcn_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-normal node features; the class is whichever of the first ten features has the largest mean over the
    nodes, so the label is learnable (the C# row's LoadSyntheticBatch)."""
    x = torch.randn(batch_size, GCN_NODES, GCN_FEATURES)
    return x.to(device), x[:, :, :10].sum(dim=1).argmax(dim=1).to(device)


class GraphConv(nn.Module):
    """Kipf & Welling's layer as AiDotNet's GraphConvolutionalLayer computes it: A_hat (X W) + b. The bias is added
    after propagation (nn.Linear inside the product would propagate it too)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return adj @ (x @ self.weight) + self.bias


class GCNClassifier(nn.Module):
    """2x GCN(16->64->64, ReLU) over one fixed 32-node graph, mean over nodes, Linear(64, 10). Input [B, 32, 16]."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("adj", kipf_graph())
        self.gc1 = GraphConv(GCN_FEATURES, 64)
        self.gc2 = GraphConv(64, 64)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gc1(x, self.adj))
        h = torch.relu(self.gc2(h, self.adj))
        return self.head(h.mean(dim=1))


DDPM_IMAGE, DDPM_CHANNELS, DDPM_STEPS, DDPM_INFERENCE_T = 32, 3, 1000, 500


class DDPMResBlock(nn.Module):
    """AiDotNet's DiffusionResBlock: GroupNorm(32, eps 1e-5) -> SiLU -> Conv3x3, plus Linear(SiLU(temb)) per channel,
    then GroupNorm -> SiLU -> Conv3x3, plus the input (through a 1x1 conv when the width changes). No dropout."""

    def __init__(self, in_ch: int, out_ch: int, temb: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-5)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(temb, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-5)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class DDPMAttention(nn.Module):
    """AiDotNet's DiffusionAttention as the U-Net builds it (residualGroupNorm): GroupNorm(32), multi-head
    self-attention over the H*W positions with bias-free Q/K/V and a biased output projection (its
    FlashAttentionLayer), then the input added back. Ho et al. 2020's attention block."""

    def __init__(self, ch: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, ch, eps=1e-5)
        self.q = nn.Linear(ch, ch, bias=False)
        self.k = nn.Linear(ch, ch, bias=False)
        self.v = nn.Linear(ch, ch, bias=False)
        self.o = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, h * w, self.heads, c // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(split(self.q(seq)), split(self.k(seq)), split(self.v(seq)))
        out = self.o(out.transpose(1, 2).reshape(b, h * w, c))
        return x + out.reshape(b, h, w, c).permute(0, 3, 1, 2)


class DDPMUNet(nn.Module):
    """The C# "ddpm" row's denoiser as AiDotNet's UNetNoisePredictor builds it for DDPMModel: 128 base channels,
    multipliers [1, 2, 2, 2], two residual blocks per level, attention (4 heads) at level 1 (16x16) and in the middle;
    a sinusoidal timestep embedding of width 512 through two Linear+ReLU; stride-2 conv down, ConvTranspose(4, 2, 1)
    up; each decoder residual block concatenates one encoder skip."""

    def __init__(self, base: int = 128, mults: tuple[int, ...] = (1, 2, 2, 2), blocks: int = 2,
                 attention_levels: tuple[int, ...] = (1,), heads: int = 4) -> None:
        super().__init__()
        self.temb_dim = base * 4
        self.time1 = nn.Linear(self.temb_dim, self.temb_dim)
        self.time2 = nn.Linear(self.temb_dim, self.temb_dim)
        self.inp = nn.Conv2d(DDPM_CHANNELS, base, 3, padding=1)
        self.encoder = nn.ModuleList()
        ch = base
        for level, mult in enumerate(mults):
            out = base * mult
            for _ in range(blocks):
                block = nn.ModuleDict({"res": DDPMResBlock(ch, out, self.temb_dim)})
                if level in attention_levels:
                    block["attn"] = DDPMAttention(out, heads)
                self.encoder.append(block)
                ch = out
            if level < len(mults) - 1:
                self.encoder.append(nn.ModuleDict({"down": nn.Conv2d(out, out, 3, stride=2, padding=1)}))
        self.mid_res1 = DDPMResBlock(ch, ch, self.temb_dim)
        self.mid_attn = DDPMAttention(ch, heads)
        self.mid_res2 = DDPMResBlock(ch, ch, self.temb_dim)
        self.decoder = nn.ModuleList()
        for level in reversed(range(len(mults))):
            out = base * mults[level]
            for _ in range(blocks):
                block = nn.ModuleDict({"res": DDPMResBlock(ch + out, out, self.temb_dim)})
                if level in attention_levels:
                    block["attn"] = DDPMAttention(out, heads)
                self.decoder.append(block)
                ch = out
            if level > 0:
                self.decoder.append(nn.ModuleDict({"up": nn.ConvTranspose2d(out, out, 4, stride=2, padding=1)}))
        self.out = nn.Conv2d(base * mults[0], DDPM_CHANNELS, 3, padding=1)
        betas = torch.linspace(1e-4, 0.02, DDPM_STEPS, dtype=torch.float64)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("sqrt_ab", alpha_bar.sqrt().float())
        self.register_buffer("sqrt_1m_ab", (1 - alpha_bar).sqrt().float())
        self._init_like_aidotnet()

    # Ho et al. 2020: Adam at 2e-4 with the gradient norm clipped to 1 (no weight decay). The C# row uses the same.
    optimizer_settings = {"lr": 2e-4, "weight_decay": 0.0}

    @torch.no_grad()
    def _init_like_aidotnet(self) -> None:
        """The initial distributions AiDotNet's layers draw, so both sides start the same network and train the same
        way. Convolutions: U(+-sqrt(3 / fan_in)) (Kaiming, gain 1), zero bias. Transposed convolutions and the
        residual blocks' time projections: Xavier normal truncated at 2 sigma, zero bias. The time MLP (ReLU): He
        normal truncated at 2 sigma. Attention Q/K/V: Xavier uniform. Zero (as Ho et al. and LDM): each residual
        block's second conv, the attention output projection and the output conv."""
        def trunc(weight: torch.Tensor, std: float) -> None:
            nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                bound = math.sqrt(3.0 / (module.in_channels * module.kernel_size[0] * module.kernel_size[1]))
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                trunc(module.weight, math.sqrt(2.0 / (fan_in + fan_out)))
                nn.init.zeros_(module.bias)
            elif isinstance(module, DDPMResBlock):
                trunc(module.time.weight, math.sqrt(2.0 / (module.time.in_features + module.time.out_features)))
                nn.init.zeros_(module.time.bias)
            elif isinstance(module, DDPMAttention):
                for projection in (module.q, module.k, module.v):
                    nn.init.xavier_uniform_(projection.weight)
        for linear in (self.time1, self.time2):
            trunc(linear.weight, math.sqrt(2.0 / linear.in_features))
            nn.init.zeros_(linear.bias)
        for module in self.modules():
            if isinstance(module, DDPMResBlock):
                nn.init.zeros_(module.conv2.weight)
                nn.init.zeros_(module.conv2.bias)
            elif isinstance(module, DDPMAttention):
                nn.init.zeros_(module.o.weight)
                nn.init.zeros_(module.o.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """NoisePredictorBase.GetTimestepEmbedding: [sin, cos] of t * exp(-i ln(10000) / (half - 1))."""
        half = self.temb_dim // 2
        freqs = torch.exp(-math.log(10000.0) / (half - 1) * torch.arange(half, device=t.device, dtype=torch.float64))
        angles = t.double()[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1).float()

    def denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = F.relu(self.time2(F.relu(self.time1(self.timestep_embedding(t)))))
        h = self.inp(x)
        skips = []
        for block in self.encoder:
            if "down" in block:
                h = block["down"](h)
                continue
            h = block["res"](h, temb)
            if "attn" in block:
                h = block["attn"](h)
            skips.append(h)
        h = self.mid_res2(self.mid_attn(self.mid_res1(h, temb)), temb)
        for block in self.decoder:
            if "up" in block:
                h = block["up"](h)
                continue
            h = block["res"](torch.cat([h, skips.pop()], dim=1), temb)
            if "attn" in block:
                h = block["attn"](h)
        return self.out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: one denoiser evaluation of the batch at timestep 500 (the C# row's Forward)."""
        return self.denoise(x, torch.full((x.shape[0],), DDPM_INFERENCE_T, device=x.device, dtype=torch.long))

    def training_loss(self, x0: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """Ho et al. 2020 Algorithm 1: a timestep and Gaussian noise per image, MSE against the noise."""
        t = torch.randint(0, DDPM_STEPS, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.sqrt_ab[t][:, None, None, None] * x0 + self.sqrt_1m_ab[t][:, None, None, None] * noise
        return F.mse_loss(self.denoise(xt, t), noise)

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean images uniform in [-1, 1] (the C# row's LoadSyntheticBatch); the label half is unused."""
        x = (torch.rand(batch_size, DDPM_CHANNELS, DDPM_IMAGE, DDPM_IMAGE) * 2 - 1).to(device)
        return x, x


WHISPER_MELS, WHISPER_FRAMES, WHISPER_TOKENS, WHISPER_VOCAB, WHISPER_SOT = 80, 3000, 16, 51865, 50258


class WhisperTiny(nn.Module):
    """The "whisper-tiny" row: transformers' WhisperForConditionalGeneration at the whisper-tiny configuration (width
    384, 4 + 4 layers, 6 heads, FFN 1536), randomly initialized. A training step is teacher-forced cross-entropy over a
    16-token transcript of a 30-second log-mel clip (labels shifted right behind <|startoftranscript|>, as AiDotNet's
    WhisperModel does); inference is the encoder plus one decoder step from the start token. AiDotNet's whisper-tiny
    has 9,216 fewer parameters: the q and v attention biases its attention layers do not have."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import WhisperConfig, WhisperForConditionalGeneration

        config = WhisperConfig(
            vocab_size=WHISPER_VOCAB, num_mel_bins=WHISPER_MELS, d_model=384, encoder_layers=4, decoder_layers=4,
            encoder_attention_heads=6, decoder_attention_heads=6, encoder_ffn_dim=1536, decoder_ffn_dim=1536,
            max_source_positions=WHISPER_FRAMES // 2, max_target_positions=448, dropout=0.0,
            decoder_start_token_id=WHISPER_SOT, pad_token_id=50257)
        self.model = WhisperForConditionalGeneration(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = torch.full((x.shape[0], 1), WHISPER_SOT, dtype=torch.long, device=x.device)
        return self.model(input_features=x, decoder_input_ids=start).logits

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(input_features=x, labels=y).loss

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-mel features in [-1, 1] (Whisper's normalized range) and random transcript tokens."""
        x = (torch.rand(batch_size, WHISPER_MELS, WHISPER_FRAMES) * 2 - 1).to(device)
        y = torch.randint(0, WHISPER_SOT, (batch_size, WHISPER_TOKENS)).to(device)
        return x, y


DETECTION_IMAGE, DETECTION_CLASSES, DETECTION_BOXES = 320, 90, 2


class FasterRcnnR50Fpn(nn.Module):
    """The "faster-rcnn" row: torchvision's fasterrcnn_resnet50_fpn with no pretrained weights (so batch normalization
    trains, as in AiDotNet's backbone), 90 classes plus background, images fixed at 320x320. A training step sums the
    RPN and RoI-head losses for one image with two boxes; inference is the full detect pipeline in eval mode. The
    optimizer is SGD at 1e-3 with momentum 0.9, AiDotNet FasterRCNN's paper recipe."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=DETECTION_CLASSES + 1,
            min_size=DETECTION_IMAGE, max_size=DETECTION_IMAGE)

    def make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model(list(x))
        finally:
            self.model.train(was_training)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.train()
        targets = [{"boxes": boxes[:, 1:], "labels": boxes[:, 0].long() + 1} for boxes in y]
        return sum(self.model(list(x), targets).values())

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Images uniform in [0, 1] and, per image, rows of (class, x1, y1, x2, y2) in pixels."""
        x = torch.rand(batch_size, 3, DETECTION_IMAGE, DETECTION_IMAGE).to(device)
        centers = torch.rand(batch_size, DETECTION_BOXES, 2) * 0.5 + 0.25
        sizes = torch.rand(batch_size, DETECTION_BOXES, 2) * 0.3 + 0.1
        corners = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=2) * DETECTION_IMAGE
        classes = torch.randint(0, DETECTION_CLASSES, (batch_size, DETECTION_BOXES, 1)).float()
        return x, torch.cat([classes, corners], dim=2).to(device)


# ---------------------------------------------------------------------------------------------------------------------
# Time-series forecasting rows (the C# side is TimeSeriesRows.cs). The ETT protocol the transformer papers report: a
# 96-step lookback, a 24-step horizon, 7 channels (N-BEATS is univariate). Each twin is the official architecture at the
# paper's default sizes, written out here so it needs no repository checkout, and built layer for layer as AiDotNet's
# shipped forecaster builds it (equal parameter counts). Neither side feeds time-feature ("timeF") marks: AiDotNet's
# tensor-interface Informer/Autoformer take the history alone, so both run the official networks without the temporal
# embedding. Optimizer: each paper's Adam (AdamW, no weight decay), with the harness's global-norm clip of 1.
# ---------------------------------------------------------------------------------------------------------------------

TS_LOOKBACK, TS_HORIZON, TS_CHANNELS, TS_LABEL = 96, 24, 7, 48


def forecast_batch(batch_size: int, channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """A noisy daily cycle 1 + 0.5 sin(2 pi (t + phase) / 24) + 0.1 N(0, 1), a random phase per series: history
    [B, 96, C] and its continuation [B, 24, C] (C = 1 is squeezed). The C# rows draw the same distribution."""
    steps = TS_LOOKBACK + TS_HORIZON
    phase = torch.rand(batch_size, 1, channels) * 24.0
    t = torch.arange(steps, dtype=torch.float32).reshape(1, steps, 1)
    series = 1.0 + 0.5 * torch.sin(2 * math.pi * (t + phase) / 24.0) + 0.1 * torch.randn(batch_size, steps, channels)
    x, y = series[:, :TS_LOOKBACK], series[:, TS_LOOKBACK:]
    if channels == 1:
        x, y = x.squeeze(-1), y.squeeze(-1)
    return x.contiguous().to(device), y.contiguous().to(device)


class CircularTokenConv(nn.Module):
    """TokenEmbedding: Conv1d(c_in, d_model, 3, padding 1, circular) over time, Kaiming-normal weights."""

    def __init__(self, channels: int, d_model: int, bias: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, d_model, 3, padding=1, padding_mode="circular", bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


def sinusoid_positions(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    table = torch.zeros(length, d_model)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table.unsqueeze(0)


class InformerProbAttention(nn.Module):
    """Informer2020 models/attn.py ProbAttention: sample U = c ln L_K keys per query, let the top u = c ln L_Q queries by
    the sparsity measurement attend, give the rest mean(V) (masked: cumsum(V)). No attention dropout (as the reference)."""

    def __init__(self, mask_flag: bool, factor: int) -> None:
        super().__init__()
        self.mask_flag, self.factor = mask_flag, factor

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, lq, h, e = q.shape
        lk = k.shape[1]
        q, k, v = q.transpose(2, 1), k.transpose(2, 1), v.transpose(2, 1)                          # [B, H, L, E]
        sample_k = min(self.factor * int(math.ceil(math.log(lk))), lk)
        n_top = min(self.factor * int(math.ceil(math.log(lq))), lq)
        index_sample = torch.randint(lk, (lq, sample_k), device=q.device)
        rows = torch.arange(lq, device=q.device).unsqueeze(1)
        k_sample = k.unsqueeze(-3).expand(b, h, lq, lk, e)[:, :, rows, index_sample, :]
        qk_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze(-2)
        sparsity = qk_sample.max(-1)[0] - torch.div(qk_sample.sum(-1), lk)
        top = sparsity.topk(n_top, sorted=False)[1]                                                # [B, H, u]
        bi = torch.arange(b, device=q.device)[:, None, None]
        hi = torch.arange(h, device=q.device)[None, :, None]
        scores = torch.matmul(q[bi, hi, top, :], k.transpose(-2, -1)) / math.sqrt(e)               # [B, H, u, L_K]
        if self.mask_flag:
            context = v.cumsum(dim=-2)
            causal = torch.ones(lq, lk, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(causal[None, None].expand(b, h, lq, lk)[bi, hi, top, :], float("-inf"))
        else:
            context = v.mean(dim=-2, keepdim=True).expand(b, h, lq, e).clone()
        context[bi, hi, top, :] = torch.matmul(torch.softmax(scores, dim=-1), v).type_as(context)
        return context.transpose(2, 1).contiguous()                                                # [B, L, H, E]


class InformerFullAttention(nn.Module):
    """models/attn.py FullAttention (unmasked, as the cross-attention uses it) with attention dropout."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        weights = self.dropout(torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1))
        return torch.einsum("bhls,bshd->blhd", weights, v).contiguous()


class InformerAttentionLayer(nn.Module):
    def __init__(self, inner: nn.Module, d_model: int, heads: int, mix: bool) -> None:
        super().__init__()
        self.inner, self.heads, self.mix = inner, heads, mix
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        b, lq, _ = queries.shape
        s, h = keys.shape[1], self.heads
        out = self.inner(self.query_projection(queries).view(b, lq, h, -1), self.key_projection(keys).view(b, s, h, -1),
                         self.value_projection(keys).view(b, s, h, -1))
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        return self.out_projection(out.view(b, lq, -1))


class PointwiseFeedForward(nn.Module):
    """The kernel-1 Conv1d feed-forward of the Encoder/DecoderLayer, as Linear maps (the same parameters), GELU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Linear(d_model, d_ff, bias=bias)
        self.conv2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv2(self.dropout(F.gelu(self.conv1(x)))))


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, factor: int, dropout: float) -> None:
        super().__init__()
        self.attention = InformerAttentionLayer(InformerProbAttention(False, factor), d_model, heads, mix=False)
        self.ff = PointwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, x)))
        return self.norm2(x + self.ff(x))


class InformerDistil(nn.Module):
    """models/encoder.py ConvLayer: circular Conv1d(k 3), BatchNorm1d, ELU, MaxPool1d(3, 2, 1)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.downConv = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.elu(self.norm(self.downConv(x.permute(0, 2, 1))))).transpose(1, 2)


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, factor: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = InformerAttentionLayer(InformerProbAttention(True, factor), d_model, heads, mix=True)
        self.cross_attention = InformerAttentionLayer(InformerFullAttention(dropout), d_model, heads, mix=False)
        self.ff = PointwiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory)))
        return self.norm3(x + self.ff(x))


class InformerForecaster(nn.Module):
    """The "informer" row: zhouhaoyi/Informer2020 models/model.py Informer at the paper's ETTh1 setting (d 512, 8 heads,
    e_layers 2 with distilling, d_layers 1, d_ff 2048, factor 5, ProbSparse attention, mix, GELU, dropout 0.05, 48
    start tokens), without the time-feature embedding (see above). The decoder reads [last 48 steps; 24 zeros]."""

    regression = True
    optimizer_settings = {"lr": 1e-4, "weight_decay": 0.0}

    def __init__(self, channels: int = TS_CHANNELS, d_model: int = 512, heads: int = 8, e_layers: int = 2,
                 d_layers: int = 1, d_ff: int = 2048, factor: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.enc_embedding = CircularTokenConv(channels, d_model, bias=True)
        self.dec_embedding = CircularTokenConv(channels, d_model, bias=True)
        self.register_buffer("enc_positions", sinusoid_positions(TS_LOOKBACK, d_model), persistent=False)
        self.register_buffer("dec_positions", sinusoid_positions(TS_LABEL + TS_HORIZON, d_model), persistent=False)
        self.encoder_layers = nn.ModuleList(InformerEncoderLayer(d_model, heads, d_ff, factor, dropout) for _ in range(e_layers))
        self.distil = nn.ModuleList(InformerDistil(d_model) for _ in range(e_layers - 1))
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList(InformerDecoderLayer(d_model, heads, d_ff, factor, dropout) for _ in range(d_layers))
        self.decoder_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dec_in = torch.cat([x[:, -TS_LABEL:, :], torch.zeros(x.shape[0], TS_HORIZON, x.shape[2], device=x.device)], dim=1)
        h = self.dropout(self.enc_embedding(x) + self.enc_positions)
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i < len(self.distil):
                h = self.distil[i](h)
        memory = self.encoder_norm(h)
        y = self.dropout(self.dec_embedding(dec_in) + self.dec_positions)
        for layer in self.decoder_layers:
            y = layer(y, memory)
        return self.projection(self.decoder_norm(y))[:, -TS_HORIZON:, :]

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return forecast_batch(batch_size, TS_CHANNELS, device)


class AutoformerDecomposition(nn.Module):
    """layers/Autoformer_EncDec.py series_decomp: the trend is AvgPool1d(k, 1) over the ends replicated (k - 1) / 2
    times; returns (seasonal, trend)."""

    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half = (self.kernel - 1) // 2
        padded = torch.cat([x[:, 0:1, :].repeat(1, half, 1), x, x[:, -1:, :].repeat(1, half, 1)], dim=1)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class AutoformerSeasonalNorm(nn.Module):
    """my_Layernorm: LayerNorm, then the mean over time subtracted."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.layernorm(x)
        return x_hat - torch.mean(x_hat, dim=1, keepdim=True)


class AutoformerCorrelationLayer(nn.Module):
    """layers/AutoCorrelation.py AutoCorrelationLayer + AutoCorrelation: the FFT autocorrelation of queries and keys,
    the top k = factor ln L lags, and the softmax-weighted sum of V rolled by each lag. Training shares the batch's
    strongest lags (time_delay_agg_training), inference picks each series' own (time_delay_agg_inference). The
    reference's training aggregation rolls V by a host integer per lag (a device sync and a torch.compile graph break);
    here the same rolls are one gather, which computes the identical function and can only help PyTorch."""

    def __init__(self, d_model: int, heads: int, factor: int) -> None:
        super().__init__()
        self.heads, self.factor = heads, factor
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        b, length, _ = queries.shape
        s, h = keys.shape[1], self.heads
        q = self.query_projection(queries).view(b, length, h, -1)
        k = self.key_projection(keys).view(b, s, h, -1)
        v = self.value_projection(keys).view(b, s, h, -1)
        if length > s:
            zeros = torch.zeros_like(q[:, : (length - s), :])
            v, k = torch.cat([v, zeros], dim=1), torch.cat([k, zeros], dim=1)
        else:
            v, k = v[:, :length], k[:, :length]
        q_fft = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=length, dim=-1)                       # [B, H, E, L]
        values = v.permute(0, 2, 3, 1).contiguous()                                                # [B, H, E, L]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)                                    # [B, L]
        steps = torch.arange(length, device=values.device)
        if self.training:
            index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]                    # [k] shared lags
            weights = torch.softmax(mean_value[:, index], dim=-1)                                  # [B, k]
            positions = (steps.unsqueeze(0) + index.unsqueeze(1)) % length                         # [k, L]
            rolled = values[..., positions]                                                        # [B, H, E, k, L]
            agg = (rolled * weights[:, None, None, :, None]).sum(dim=3)
        else:
            weights, delay = torch.topk(mean_value, top_k, dim=-1)                                 # [B, k] per series
            weights = torch.softmax(weights, dim=-1)
            doubled = values.repeat(1, 1, 1, 2)
            agg = torch.zeros_like(values)
            for i in range(top_k):
                index = (steps.view(1, 1, 1, length) + delay[:, i].view(b, 1, 1, 1)).expand(b, h, values.shape[2], length)
                agg = agg + torch.gather(doubled, dim=-1, index=index) * weights[:, i].view(b, 1, 1, 1)
        return self.out_projection(agg.permute(0, 3, 1, 2).reshape(b, length, -1))


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, factor: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.attention = AutoformerCorrelationLayer(d_model, heads, factor)
        self.ff = PointwiseFeedForward(d_model, d_ff, dropout, bias=False)
        self.decomp1, self.decomp2 = AutoformerDecomposition(kernel), AutoformerDecomposition(kernel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.decomp1(x + self.dropout(self.attention(x, x)))
        x, _ = self.decomp2(x + self.ff(x))
        return x


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, factor: int, kernel: int, dropout: float, c_out: int) -> None:
        super().__init__()
        self.self_attention = AutoformerCorrelationLayer(d_model, heads, factor)
        self.cross_attention = AutoformerCorrelationLayer(d_model, heads, factor)
        self.ff = PointwiseFeedForward(d_model, d_ff, dropout, bias=False)
        self.decomp1 = AutoformerDecomposition(kernel)
        self.decomp2 = AutoformerDecomposition(kernel)
        self.decomp3 = AutoformerDecomposition(kernel)
        self.projection = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, trend1 = self.decomp1(x + self.dropout(self.self_attention(x, x)))
        x, trend2 = self.decomp2(x + self.dropout(self.cross_attention(x, memory)))
        x, trend3 = self.decomp3(x + self.ff(x))
        residual = trend1 + trend2 + trend3
        return x, self.projection(residual.permute(0, 2, 1)).transpose(1, 2)


class AutoformerForecaster(nn.Module):
    """The "autoformer" row: thuml/Autoformer models/Autoformer.py at the paper's setting (d 512, 8 heads, e_layers 2,
    d_layers 1, d_ff 2048, moving average 25, factor 1, GELU, dropout 0.05, 48 start tokens), without the time-feature
    embedding (see above). The decoder starts from the history's decomposition: [last 48 seasonal steps; zeros] and
    [last 48 trend steps; the history mean]."""

    regression = True
    optimizer_settings = {"lr": 1e-4, "weight_decay": 0.0}

    def __init__(self, channels: int = TS_CHANNELS, d_model: int = 512, heads: int = 8, e_layers: int = 2,
                 d_layers: int = 1, d_ff: int = 2048, kernel: int = 25, factor: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.decomp = AutoformerDecomposition(kernel)
        self.enc_embedding = CircularTokenConv(channels, d_model, bias=False)
        self.dec_embedding = CircularTokenConv(channels, d_model, bias=False)
        self.encoder_layers = nn.ModuleList(
            AutoformerEncoderLayer(d_model, heads, d_ff, factor, kernel, dropout) for _ in range(e_layers))
        self.encoder_norm = AutoformerSeasonalNorm(d_model)
        self.decoder_layers = nn.ModuleList(
            AutoformerDecoderLayer(d_model, heads, d_ff, factor, kernel, dropout, channels) for _ in range(d_layers))
        self.decoder_norm = AutoformerSeasonalNorm(d_model)
        self.projection = nn.Linear(d_model, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=1, keepdim=True).repeat(1, TS_HORIZON, 1)
        zeros = torch.zeros(x.shape[0], TS_HORIZON, x.shape[2], device=x.device)
        seasonal, trend = self.decomp(x)
        trend = torch.cat([trend[:, -TS_LABEL:, :], mean], dim=1)
        seasonal = torch.cat([seasonal[:, -TS_LABEL:, :], zeros], dim=1)
        h = self.dropout(self.enc_embedding(x))
        for layer in self.encoder_layers:
            h = layer(h)
        memory = self.encoder_norm(h)
        y = self.dropout(self.dec_embedding(seasonal))
        for layer in self.decoder_layers:
            y, residual_trend = layer(y, memory)
            trend = trend + residual_trend
        return (trend + self.projection(self.decoder_norm(y)))[:, -TS_HORIZON:, :]

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return forecast_batch(batch_size, TS_CHANNELS, device)


class DeepARForecaster(nn.Module):
    """The "deepar" row: DeepAR (Salinas et al. 2020) as AiDotNet's DeepAR builds it: each of the 7 channels is a series
    (rows [B*7]) with a 20-wide embedding of its index, nn.LSTM(1 + 20, 40, 3 layers, dropout 0.1) over
    [z_{t-1} / nu, embedding], and a Gaussian head (mu = nu * a, sigma = nu * softplus(b)) with nu = 1 + the mean of the
    conditioning range. Training minimizes the Gaussian negative log-likelihood of every step of the 96 + 24 window
    (z_0 = 0 in front); a forecast is the per-step median of 200 sample paths drawn ancestrally over the horizon. The
    loss comes from forward(x, y) so that --compile compiles it (a training_loss method would run eager)."""

    loss_from_forward = True
    optimizer_settings = {"lr": 1e-3, "weight_decay": 0.0}

    def __init__(self, channels: int = TS_CHANNELS, hidden: int = 40, layers: int = 3, embedding: int = 20,
                 dropout: float = 0.1, samples: int = 200) -> None:
        super().__init__()
        self.channels, self.samples = channels, samples
        self.embedding = nn.Embedding(channels, embedding)
        self.lstm = nn.LSTM(1 + embedding, hidden, layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    @staticmethod
    def _rows(values: torch.Tensor) -> torch.Tensor:
        return values.permute(0, 2, 1).reshape(-1, values.shape[1])                                # [B*C, steps]

    @staticmethod
    def _inputs(z: torch.Tensor, nu: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([(z / nu).unsqueeze(-1), emb.unsqueeze(1).expand(-1, z.shape[1], -1)], dim=-1)

    def _gaussian(self, top: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(top)
        return raw[..., 0] * nu, F.softplus(raw[..., 1]) * nu

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        history = self._rows(x)                                                                    # [N, L]
        emb = self.embedding(torch.arange(self.channels, device=x.device).repeat(b))               # [N, E]
        nu = 1.0 + history.mean(dim=1, keepdim=True)
        zero = torch.zeros(history.shape[0], 1, device=x.device)
        if y is not None:
            z = torch.cat([zero, history, self._rows(y)], dim=1)                                   # [N, L + H + 1]
            top, _ = self.lstm(self._inputs(z[:, :-1], nu, emb))
            mu, sigma = self._gaussian(top, nu)
            standardized = (z[:, 1:] - mu) / sigma
            return (torch.log(sigma) + 0.5 * standardized * standardized + 0.5 * math.log(2 * math.pi)).mean()
        # The conditioning range once per series; its state is copied to every sample path.
        _, (h, c) = self.lstm(self._inputs(torch.cat([zero, history[:, :-1]], dim=1), nu, emb))
        h, c = h.repeat_interleave(self.samples, dim=1), c.repeat_interleave(self.samples, dim=1)
        nu_rows = nu.repeat_interleave(self.samples, dim=0)
        emb_rows = emb.repeat_interleave(self.samples, dim=0)
        last = history[:, -1:].repeat_interleave(self.samples, dim=0)                              # [N*S, 1]
        draws = []
        for _ in range(TS_HORIZON):
            top, (h, c) = self.lstm(self._inputs(last, nu_rows, emb_rows), (h, c))
            mu, sigma = self._gaussian(top, nu_rows)
            last = mu + sigma * torch.randn_like(mu)
            draws.append(last)
        paths = torch.cat(draws, dim=1).view(-1, self.samples, TS_HORIZON)                         # [N, S, H]
        # AiDotNet's quantile: the floor(0.5 * S)-th order statistic.
        median = paths.sort(dim=1)[0][:, self.samples // 2, :]
        return median.view(b, self.channels, TS_HORIZON).permute(0, 2, 1)

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return forecast_batch(batch_size, TS_CHANNELS, device)


class NBeatsForecaster(nn.Module):
    """The "nbeats" row: N-BEATS-G (Oreshkin et al. 2020; ElementAI's models/nbeats.py GenericBasis) at the paper's
    generic setting, 30 stacks of 1 block, 4 ReLU layers of 512, univariate 96 -> 24. A block's theta Linear(512,
    96 + 24) split into backcast and forecast is written as the two Linear maps AiDotNet builds (the same parameters).
    AiDotNet's NBEATSFinance adds a final Linear(24, 24) over the summed forecast; the twin has it too (600 parameters)."""

    regression = True
    optimizer_settings = {"lr": 1e-3, "weight_decay": 0.0}

    def __init__(self, stacks: int = 30, layers: int = 4, width: int = 512) -> None:
        super().__init__()

        def block() -> nn.ModuleDict:
            fc: list[nn.Module] = [nn.Linear(TS_LOOKBACK, width), nn.ReLU()]
            for _ in range(layers - 1):
                fc += [nn.Linear(width, width), nn.ReLU()]
            return nn.ModuleDict({"fc": nn.Sequential(*fc), "backcast": nn.Linear(width, TS_LOOKBACK),
                                  "forecast": nn.Linear(width, TS_HORIZON)})

        self.blocks = nn.ModuleList(block() for _ in range(stacks))
        self.output_projection = nn.Linear(TS_HORIZON, TS_HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, forecast = x, None
        for blk in self.blocks:
            hidden = blk["fc"](residual)
            residual = residual - blk["backcast"](hidden)
            step = blk["forecast"](hidden)
            forecast = step if forecast is None else forecast + step
        return self.output_projection(forecast)

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return forecast_batch(batch_size, 1, device)


TIME_SERIES_MODELS = {"informer": (InformerForecaster, (TS_LOOKBACK, TS_CHANNELS)),
                      "autoformer": (AutoformerForecaster, (TS_LOOKBACK, TS_CHANNELS)),
                      "deepar": (DeepARForecaster, (TS_LOOKBACK, TS_CHANNELS)),
                      "nbeats": (NBeatsForecaster, (TS_LOOKBACK,))}


class TfSamePad(nn.Module):
    """TensorFlow "same" padding ahead of a strided convolution: total = max((ceil(n / s) - 1) * s + k - n, 0), the
    smaller half before. AiDotNet's EfficientNetTrunk.SamePad pads the same way (stride 1 convolutions pad k // 2)."""

    def __init__(self, kernel: int, stride: int) -> None:
        super().__init__()
        self.kernel, self.stride = kernel, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads: list[int] = []
        for n in (x.shape[3], x.shape[2]):  # F.pad takes the last axis first
            total = max((math.ceil(n / self.stride) - 1) * self.stride + self.kernel - n, 0)
            pads += [total // 2, total - total // 2]
        return F.pad(x, pads)


def efficientnet_conv(inputs: int, outputs: int, kernel: int, stride: int, groups: int = 1) -> nn.Module:
    """A bias-free convolution with AiDotNet's padding: k // 2 at stride 1, TensorFlow "same" at stride 2."""
    if stride == 1:
        return nn.Conv2d(inputs, outputs, kernel, 1, kernel // 2, groups=groups, bias=False)
    return nn.Sequential(TfSamePad(kernel, stride), nn.Conv2d(inputs, outputs, kernel, stride, 0, groups=groups, bias=False))


def efficientnet_norm(channels: int) -> nn.BatchNorm2d:
    """AiDotNet's trunk BatchNorm: eps 1e-3 and TensorFlow momentum 0.99 (PyTorch momentum 0.01)."""
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class MBConv(nn.Module):
    """EfficientNet's mobile inverted bottleneck as AiDotNet's EfficientNetTrunk runs it (and torchvision's MBConv):
    1x1 expand + BN + SiLU (when the ratio is not 1), depthwise k x k + BN + SiLU, squeeze-excitation (mean pool,
    Linear to max(1, inputs // 4), SiLU, Linear back, sigmoid gate), 1x1 project + BN, and a residual with
    per-sample stochastic depth when the shape is kept."""

    def __init__(self, inputs: int, outputs: int, expand: int, kernel: int, stride: int, drop_rate: float) -> None:
        super().__init__()
        hidden = inputs * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [efficientnet_conv(inputs, hidden, 1, 1), efficientnet_norm(hidden), nn.SiLU()]
        layers += [efficientnet_conv(hidden, hidden, kernel, stride, groups=hidden), efficientnet_norm(hidden), nn.SiLU()]
        self.features = nn.Sequential(*layers)
        squeezed = max(1, inputs // 4)
        self.reduce, self.excite = nn.Linear(hidden, squeezed), nn.Linear(squeezed, hidden)
        self.project = nn.Sequential(efficientnet_conv(hidden, outputs, 1, 1), efficientnet_norm(outputs))
        self.residual = stride == 1 and inputs == outputs
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        gate = torch.sigmoid(self.excite(F.silu(self.reduce(h.mean(dim=(2, 3))))))
        h = self.project(h * gate[:, :, None, None])
        if not self.residual:
            return h
        if self.training and self.drop_rate > 0:
            keep = (torch.rand(h.shape[0], 1, 1, 1, device=h.device) >= self.drop_rate).to(h.dtype)
            h = h * keep / (1.0 - self.drop_rate)
        return h + x


class EfficientNetB0(nn.Module):
    """The "efficientnet-b0" row: EfficientNet-B0 (Tan & Le 2019; width and depth 1.0) at CIFAR [3, 32, 32], 10
    classes, as AiDotNet's EfficientNetNetwork builds it: stem 3x3/2 to 32, the seven MBConv stages, 1x1 head to 1280,
    global mean pool, dropout 0.1 (AiDotNet's Custom-variant rate), Linear(1280, 10). Drop-connect rises linearly to
    0.2 over the 16 blocks. Same layout and trainable parameter count as torchvision's efficientnet_b0 at 10 classes
    (4,020,358); defined here rather than taken from torchvision so the BatchNorm settings and the TensorFlow "same"
    padding of the strided convolutions match AiDotNet's trunk exactly."""

    STAGES = ((1, 3, 1, 32, 16, 1), (6, 3, 2, 16, 24, 2), (6, 5, 2, 24, 40, 2), (6, 3, 2, 40, 80, 3),
              (6, 5, 1, 80, 112, 3), (6, 5, 2, 112, 192, 4), (6, 3, 1, 192, 320, 1))

    def __init__(self, classes: int = 10, dropout: float = 0.1, drop_connect: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(efficientnet_conv(3, 32, 3, 2), efficientnet_norm(32), nn.SiLU())
        total = sum(stage[5] for stage in self.STAGES)
        blocks: list[nn.Module] = []
        for expand, kernel, stride, inputs, outputs, count in self.STAGES:
            for b in range(count):
                blocks.append(MBConv(inputs if b == 0 else outputs, outputs, expand, kernel, stride if b == 0 else 1,
                                     drop_connect * len(blocks) / total))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(efficientnet_conv(320, 1280, 1, 1), efficientnet_norm(1280), nn.SiLU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1280, classes)
        # The reference initialization (torchvision's, and the official code AiDotNet follows): convolutions normal
        # with fan-out scaling, dense weights U(+-1/sqrt(out)) with zero bias. The dense bound puts the initial loss
        # well above ln(10) on both sides.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.Linear):
                bound = 1.0 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -bound, bound)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.blocks(self.stem(x)))
        return self.classifier(self.dropout(h.mean(dim=(2, 3))))


VAE_PIXELS, VAE_HIDDEN, VAE_LATENT = 784, 400, 20


class MnistVAE(nn.Module):
    """The "vae" row: Kingma & Welling's MNIST VAE in the pytorch/examples layout, 784 -> 400 (ReLU) -> 20 means and
    20 log-variances -> reparameterized sample -> 400 (ReLU) -> 784 (sigmoid), 652,824 parameters. fc2 is AiDotNet's
    single 400 -> 40 layer (means, then log-variances); pytorch/examples' fc21/fc22 pair is the same weights. The loss
    is AiDotNet VariationalAutoencoder's: mean squared reconstruction error plus beta = 0.01 times the KL divergence
    summed over the latent and averaged over the batch. Inference decodes the latent mean."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(VAE_PIXELS, VAE_HIDDEN)
        self.fc2 = nn.Linear(VAE_HIDDEN, 2 * VAE_LATENT)
        self.fc3 = nn.Linear(VAE_LATENT, VAE_HIDDEN)
        self.fc4 = nn.Linear(VAE_HIDDEN, VAE_PIXELS)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.fc2(F.relu(self.fc1(x))).chunk(2, dim=1)
        return mean, log_variance

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.encode(x)
        return self.decode(mean)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, log_variance = self.encode(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        kl = 0.5 * (log_variance.exp() + mean * mean - 1 - log_variance).sum() / x.shape[0]
        return F.mse_loss(self.decode(z), y) + 0.01 * kl

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened 28x28 images with pixels uniform in [0, 1]; the target is the input."""
        x = torch.rand(batch_size, VAE_PIXELS).to(device)
        return x, x


GAT_NODES, GAT_FEATURES, GAT_CLASSES = 2708, 1433, 7


def cora_like_graph(nodes: int = GAT_NODES) -> torch.Tensor:
    """The C# gat row's graph: a ring plus a chord i -> i+7 from every node, undirected, with self-loops - 5,416 edges
    against Cora's 5,429. Returned as the additive attention mask AiDotNet's GraphAttentionLayer builds: 0 on an
    edge, -1e9 elsewhere."""
    adjacency = torch.eye(nodes)
    index = torch.arange(nodes)
    for step in (1, 7):
        adjacency[index, (index + step) % nodes] = 1
        adjacency[(index + step) % nodes, index] = 1
    return (1 - adjacency) * -1e9


class DenseGATLayer(nn.Module):
    """AiDotNet's GraphAttentionLayer on its dense path (and the GAT authors' reference implementation): per head,
    Wh = X W, e_ij = LeakyReLU_0.2(a_self . Wh_i + a_neigh . Wh_j) + mask_ij, softmax over j, dropout on the
    coefficients, then aggregation; heads are concatenated (hidden layer) or averaged (output layer), plus a bias."""

    def __init__(self, inputs: int, outputs: int, heads: int, concat: bool, dropout: float) -> None:
        super().__init__()
        self.outputs, self.concat, self.dropout = outputs, concat, dropout
        self.weight = nn.Parameter(torch.empty(heads, inputs, outputs))
        self.attention = nn.Parameter(torch.empty(heads, 2 * outputs))
        self.bias = nn.Parameter(torch.zeros(heads * outputs if concat else outputs))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        wh = torch.einsum("bnf,hfo->bhno", x, self.weight)                              # [B, H, N, O]
        self_scores = torch.einsum("bhno,ho->bhn", wh, self.attention[:, :self.outputs])
        neighbor_scores = torch.einsum("bhno,ho->bhn", wh, self.attention[:, self.outputs:])
        scores = F.leaky_relu(self_scores[..., :, None] + neighbor_scores[..., None, :], 0.2) + mask
        coefficients = F.dropout(torch.softmax(scores, dim=-1), self.dropout, self.training)
        out = coefficients @ wh                                                           # [B, H, N, O]
        out = out.permute(0, 2, 1, 3).flatten(2) if self.concat else out.mean(dim=1)
        return out + self.bias


class CoraGAT(nn.Module):
    """The "gat" row: Velickovic et al.'s Cora GAT as AiDotNet's GraphAttentionNetwork builds it - 1,433 features ->
    8 heads x 8 (ELU, concatenated) -> 1 head x 7 classes, attention dropout 0.6 - trained full-batch for node
    classification on the Cora-sized graph, 92,373 parameters (PyG's GATConv pair on Cora has the same count).
    AiDotNet applies the paper's 0.6 dropout to the attention coefficients only, not also to each layer's input, so
    this twin does too. One sample is the whole graph: the row runs at batch size 1."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", cora_like_graph())
        self.gat1 = DenseGATLayer(GAT_FEATURES, 8, 8, concat=True, dropout=0.6)
        self.gat2 = DenseGATLayer(64, GAT_CLASSES, 1, concat=False, dropout=0.6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gat2(F.elu(self.gat1(x, self.mask)), self.mask)

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """One graph of standard-normal node features; each node's class is the largest of its first seven features."""
        if batch_size != 1:
            raise ValueError("The gat row runs on the whole Cora-sized graph as one sample: use --batch-size 1 "
                             "--inference-batches 1.")
        x = torch.randn(1, GAT_NODES, GAT_FEATURES)
        return x.to(device), x[..., :GAT_CLASSES].argmax(dim=-1).to(device)


# --precision (main sets it): "fp32" or "bf16".
PRECISION = "fp32"

DCGAN_LATENT, DCGAN_IMAGE = 100, 64


class DCGAN64(nn.Module):
    """The "dcgan" row: Radford et al.'s 64x64 DCGAN as AiDotNet's DCGAN builds it. Generator: Linear(100, 512*4*4) +
    reshape, transposed convolutions 4x4/2 512 -> 256 -> 128 -> 64 with BatchNorm + ReLU, -> 3 with tanh.
    Discriminator: 4x4/2 convolutions 3 -> 64 (LeakyReLU 0.2) -> 128 -> 256 -> 512 (BatchNorm + LeakyReLU 0.2), then a
    4x4 convolution to one logit. Layers carry biases and N(0, 0.02) weights, as AiDotNet's do; unlike the paper (and
    pytorch/examples), AiDotNet's generator has no BatchNorm + ReLU after the projection, so this twin has none either.
    6,350,852 parameters (generator 3,584,323, discriminator 2,766,529). A training step is AiDotNet's adversarial
    iteration: the discriminator on the concatenated [real; generated] batch with BCE-with-logits, then the generator
    through the frozen eval-mode discriminator with the non-saturating loss softplus(-D(G(z))), each under Adam
    (lr 2e-4, betas 0.5 / 0.999) with the global-norm clip at 1. The reported loss is the discriminator's."""

    def __init__(self) -> None:
        super().__init__()

        def up(inputs: int, outputs: int) -> list[nn.Module]:
            return [nn.ConvTranspose2d(inputs, outputs, 4, 2, 1), nn.BatchNorm2d(outputs), nn.ReLU(True)]

        def down(inputs: int, outputs: int) -> list[nn.Module]:
            return [nn.Conv2d(inputs, outputs, 4, 2, 1), nn.BatchNorm2d(outputs), nn.LeakyReLU(0.2, True)]

        self.generator = nn.Sequential(
            nn.Linear(DCGAN_LATENT, 512 * 4 * 4), nn.Unflatten(1, (512, 4, 4)),
            *up(512, 256), *up(256, 128), *up(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True), *down(64, 128), *down(128, 256), *down(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0), nn.Flatten())
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)
        self._optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def compile_parts(self) -> None:
        """--compile: train_step calls the two networks directly, so compile them rather than only forward."""
        self.generator = torch.compile(self.generator)
        self.discriminator = torch.compile(self.discriminator)

    def train_step(self, z: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        if self._optimizers is None:
            self._optimizers = (torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
                                torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)))
        generator_optimizer, discriminator_optimizer = self._optimizers
        batch = z.shape[0]

        with torch.no_grad():
            fake = self.generator(z)
        labels = torch.cat([torch.ones(batch, 1, device=z.device), torch.zeros(batch, 1, device=z.device)])
        discriminator_loss = F.binary_cross_entropy_with_logits(self.discriminator(torch.cat([real, fake])), labels)
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        discriminator_optimizer.step()

        self.discriminator.eval()
        self.discriminator.requires_grad_(False)
        try:
            generator_loss = F.softplus(-self.discriminator(self.generator(z))).mean()
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            generator_optimizer.step()
        finally:
            self.discriminator.requires_grad_(True)
            self.discriminator.train()
        return discriminator_loss.detach()

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard-normal latents and real images uniform in [-1, 1], the tanh range."""
        z = torch.randn(batch_size, DCGAN_LATENT).to(device)
        return z, (torch.rand(batch_size, 3, DCGAN_IMAGE, DCGAN_IMAGE) * 2 - 1).to(device)


PANNS_FRAMES, PANNS_MELS, PANNS_CLASSES, PANNS_TAGS = 1001, 64, 527, 2


class PannsConvBlock(nn.Module):
    """Kong et al.'s ConvBlock: two 3x3 convolutions, each with BatchNorm + ReLU, then average pooling. The
    convolutions carry biases because AiDotNet's ConvolutionalLayer does (Kong's have none: 2 x the channel count more
    parameters per block, 8,064 in all)."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, 1, 1)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.bn2 = nn.BatchNorm2d(outputs)

    def forward(self, x: torch.Tensor, pool: int) -> torch.Tensor:
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, pool) if pool > 1 else x


class PannsCnn14(nn.Module):
    """The "panns-cnn14" row: Kong et al.'s released Cnn14 (qiuqiangkong/audioset_tagging_cnn) without its STFT/log-mel
    front end, on [B, 1, 1001 frames, 64 mels] log-mel clips: bn0 over the mel bins, six ConvBlocks 64 -> 2048 with 2x2
    average pooling in the first five and dropout 0.2 after each, mean over frequency, max + mean over time, dropout
    0.5, fc1 2048 (ReLU), dropout 0.5, 527 AudioSet logits. Trained with BCE on logits against multi-hot labels (two
    tags per clip); inference returns the sigmoid clip-wise probabilities, as AiDotNet's PANNs.Predict does. The
    trainable parameter count equals AiDotNet's (see PannsConvBlock for the convolution biases)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(PANNS_MELS)
        widths = [1, 64, 128, 256, 512, 1024, 2048]
        self.blocks = nn.ModuleList(PannsConvBlock(widths[i], widths[i + 1]) for i in range(6))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, PANNS_CLASSES)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for index, block in enumerate(self.blocks):
            x = F.dropout(block(x, 2 if index < 5 else 1), 0.2, self.training)
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.relu(self.fc1(F.dropout(x, 0.5, self.training)))
        return self.fc_audioset(F.dropout(x, 0.5, self.training))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self.logits(x), y)

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard-normal log-mel features and multi-hot labels with two random tags per clip."""
        x = torch.randn(batch_size, 1, PANNS_FRAMES, PANNS_MELS)
        y = torch.zeros(batch_size, PANNS_CLASSES)
        y.scatter_(1, torch.randint(0, PANNS_CLASSES, (batch_size, PANNS_TAGS)), 1.0)
        return x.to(device), y.to(device)


W2V_SAMPLES, W2V_TRANSCRIPT, W2V_VOCAB, W2V_HIDDEN, W2V_HEADS = 32000, 24, 34, 768, 12


class Wav2Vec2EncoderLayer(nn.Module):
    """transformers' Wav2Vec2EncoderLayer (post-norm): x = LN(x + drop(attn(x))); x = LN(x + drop(W2 drop(gelu(W1 x)))).
    Attention runs through scaled_dot_product_attention with its attention-probability dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(W2V_HIDDEN, W2V_HIDDEN)
        self.k_proj = nn.Linear(W2V_HIDDEN, W2V_HIDDEN)
        self.v_proj = nn.Linear(W2V_HIDDEN, W2V_HIDDEN)
        self.out_proj = nn.Linear(W2V_HIDDEN, W2V_HIDDEN)
        self.layer_norm = nn.LayerNorm(W2V_HIDDEN)
        self.intermediate_dense = nn.Linear(W2V_HIDDEN, 3072)
        self.output_dense = nn.Linear(3072, W2V_HIDDEN)
        self.final_layer_norm = nn.LayerNorm(W2V_HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        def heads(projection: nn.Linear) -> torch.Tensor:
            return projection(x).view(b, t, W2V_HEADS, -1).transpose(1, 2)

        context = F.scaled_dot_product_attention(heads(self.q_proj), heads(self.k_proj), heads(self.v_proj),
                                                 dropout_p=0.1 if self.training else 0.0)
        attention = self.out_proj(context.transpose(1, 2).reshape(b, t, W2V_HIDDEN))
        x = self.layer_norm(x + F.dropout(attention, 0.1, self.training))
        hidden = F.dropout(F.gelu(self.intermediate_dense(x)), 0.1, self.training)
        return self.final_layer_norm(x + F.dropout(self.output_dense(hidden), 0.1, self.training))


class Wav2Vec2BaseCtc(nn.Module):
    """The "wav2vec2-base" row: transformers' Wav2Vec2ForCTC at the BASE configuration, written out layer for layer
    (transformers is not a dependency): a 7-layer feature encoder (512 channels, kernels 10, 3, 3, 3, 3, 2, 2, strides
    5, 2, 2, 2, 2, 2, 2, no conv bias, GroupNorm(512, 512) after the first, GELU), feature projection (LayerNorm 512,
    Linear 512 -> 768), SpecAugment time masks replaced by a learned embedding (p 0.05, span 10, at least 2 spans), a
    16-group positional convolution of kernel 128 with weight norm over dim 2 (GELU), LayerNorm, dropout, 12 post-norm
    Transformer layers (12 heads, FFN 3072, dropout 0.1), final dropout 0.1 and a 34-symbol CTC head (blank 0). It is
    fine-tuned as AiDotNet's Wav2Vec2Model is: feature encoder frozen, LayerDrop 0, log-softmax + CTC per utterance
    over every frame averaged over the batch, AdamW lr 5e-5 without weight decay and the global-norm clip at 1. On
    2-second clips of 16 kHz audio (99 frames) with 24-symbol transcripts; inference returns the CTC logits
    [B, 99, 34]. Trainable parameter counts are equal."""

    optimizer_settings = {"lr": 5e-5, "weight_decay": 0.0}
    kernels, strides = (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(1 if i == 0 else 512, 512, k, s, bias=False) for i, (k, s) in enumerate(zip(self.kernels, self.strides)))
        self.conv_norm = nn.GroupNorm(512, 512)
        self.projection_norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, W2V_HIDDEN)
        self.masked_spec_embed = nn.Parameter(torch.rand(W2V_HIDDEN))
        self.pos_conv = weight_norm(nn.Conv1d(W2V_HIDDEN, W2V_HIDDEN, 128, padding=64, groups=16), name="weight", dim=2)
        self.encoder_norm = nn.LayerNorm(W2V_HIDDEN)
        self.layers = nn.ModuleList(Wav2Vec2EncoderLayer() for _ in range(12))
        self.lm_head = nn.Linear(W2V_HIDDEN, W2V_VOCAB)
        # freeze_feature_encoder(): the convolutional feature encoder does not train.
        self.conv_layers.requires_grad_(False)
        self.conv_norm.requires_grad_(False)

    def time_mask(self, batch: int, frames: int, device: torch.device) -> torch.Tensor:
        """transformers' _compute_mask_indices: per utterance floor(p * L / M + U(0, 1)) spans of M frames, at least
        two, with starts drawn without replacement."""
        span = 10
        mask = torch.zeros(batch, frames, dtype=torch.bool, device=device)
        for row in range(batch):
            spans = max(2, int(0.05 * frames / span + random.random()))
            starts = torch.randperm(frames - span + 1, device=device)[:spans]
            mask[row, (starts[:, None] + torch.arange(span, device=device)).flatten()] = True
        return mask

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        for index, conv in enumerate(self.conv_layers):
            h = conv(h)
            if index == 0:
                h = self.conv_norm(h)
            h = F.gelu(h)
        h = self.projection(self.projection_norm(h.transpose(1, 2)))
        if self.training:
            mask = self.time_mask(h.shape[0], h.shape[1], h.device)
            h = torch.where(mask[..., None], self.masked_spec_embed.to(h.dtype), h)
        position = F.gelu(self.pos_conv(h.transpose(1, 2))[:, :, :-1]).transpose(1, 2)
        h = F.dropout(self.encoder_norm(h + position), 0.1, self.training)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(F.dropout(h, 0.1, self.training))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(self.logits(x), dim=-1).transpose(0, 1)
        frames, batch = log_probs.shape[0], log_probs.shape[1]
        input_lengths = torch.full((batch,), frames, dtype=torch.long, device=x.device)
        target_lengths = torch.full((batch,), y.shape[1], dtype=torch.long, device=x.device)
        return F.ctc_loss(log_probs, y, input_lengths, target_lengths, blank=0, reduction="none").mean()

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """A random 100-400 Hz tone plus 0.5 x standard-normal noise per clip, and transcripts of random non-blank symbols."""
        t = torch.arange(W2V_SAMPLES) / 16000.0
        frequency = 100 + 300 * torch.rand(batch_size, 1)
        x = torch.sin(2 * math.pi * frequency * t) + 0.5 * torch.randn(batch_size, W2V_SAMPLES)
        y = torch.randint(1, W2V_VOCAB, (batch_size, W2V_TRANSCRIPT))
        return x.to(device), y.to(device)


HIFI_RATE, HIFI_MELS, HIFI_HOP, HIFI_FFT, HIFI_FRAMES, HIFI_SEGMENT = 22050, 80, 256, 1024, 256, 8192


def slaney_mel_basis(rate: int, fft: int, mels: int, fmin: float, fmax: float) -> torch.Tensor:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with its defaults (Slaney scale and area normalization), the
    basis the reference meldataset.mel_spectrogram uses; written out because librosa is not a dependency."""
    f_sp, min_log_hz, min_log_mel, log_step = 200.0 / 3, 1000.0, 15.0, math.log(6.4) / 27.0

    def hz_to_mel(f: float) -> float:
        return min_log_mel + math.log(f / min_log_hz) / log_step if f >= min_log_hz else f / f_sp

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return torch.where(m >= min_log_mel, min_log_hz * torch.exp(log_step * (m - min_log_mel)), f_sp * m)

    fft_freqs = torch.linspace(0, rate / 2, fft // 2 + 1, dtype=torch.float64)
    edges = mel_to_hz(torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), mels + 2, dtype=torch.float64))
    widths = edges[1:] - edges[:-1]
    ramps = edges[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    weights *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    return weights.float()


class HifiMel(nn.Module):
    """meldataset.mel_spectrogram: reflect padding of (n_fft - hop) / 2, a Hann-windowed STFT without centring,
    magnitude sqrt(re^2 + im^2 + 1e-9), the Slaney mel basis, log(max(x, 1e-5)). [B, T] -> [B, 80, frames]."""

    def __init__(self, fmax: float) -> None:
        super().__init__()
        self.register_buffer("basis", slaney_mel_basis(HIFI_RATE, HIFI_FFT, HIFI_MELS, 0.0, fmax), persistent=False)
        self.register_buffer("window", torch.hann_window(HIFI_FFT), persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        pad = (HIFI_FFT - HIFI_HOP) // 2
        y = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.view_as_real(torch.stft(y, HIFI_FFT, HIFI_HOP, HIFI_FFT, self.window, center=False,
                                             return_complex=True))
        magnitude = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return torch.log(torch.clamp(self.basis @ magnitude, min=1e-5))


def hifi_init(module: nn.Module) -> nn.Module:
    """The reference init_weights: N(0, 0.01) weights (drawn before weight norm splits them); biases keep their default."""
    nn.init.normal_(module.weight, 0.0, 0.01)
    return module


class HifiResBlock1(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm

        def conv(dilation: int) -> nn.Module:
            return weight_norm(hifi_init(nn.Conv1d(channels, channels, kernel, 1, dilation=dilation,
                                                   padding=(kernel * dilation - dilation) // 2)))

        self.convs1 = nn.ModuleList(conv(d) for d in dilations)
        self.convs2 = nn.ModuleList(conv(1) for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for first, second in zip(self.convs1, self.convs2):
            x = second(F.leaky_relu(first(F.leaky_relu(x, 0.1)), 0.1)) + x
        return x


class HifiGenerator(nn.Module):
    """The reference Generator at V1: conv_pre 80 -> 512 (7), per stage leaky ReLU 0.1 + ConvTranspose1d (rates 8, 8, 2,
    2; kernels 16, 16, 4, 4) and the mean of three ResBlock1s (kernels 3, 7, 11; dilations 1, 3, 5), leaky ReLU 0.01,
    conv_post -> 1 (7), tanh; weight norm on every convolution."""

    def __init__(self) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm

        self.conv_pre = weight_norm(nn.Conv1d(HIFI_MELS, 512, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = 512
        for rate, kernel in zip((8, 8, 2, 2), (16, 16, 4, 4)):
            self.ups.append(weight_norm(hifi_init(
                nn.ConvTranspose1d(channels, channels // 2, kernel, rate, padding=(kernel - rate) // 2))))
            channels //= 2
            for block_kernel in (3, 7, 11):
                self.resblocks.append(HifiResBlock1(channels, block_kernel, (1, 3, 5)))
        self.conv_post = weight_norm(hifi_init(nn.Conv1d(channels, 1, 7, 1, padding=3)))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for stage, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            x = (self.resblocks[stage * 3](x) + self.resblocks[stage * 3 + 1](x) + self.resblocks[stage * 3 + 2](x)) / 3
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class HifiPeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm

        self.period = period
        widths = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(widths[i], widths[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5))
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class HifiScaleDiscriminator(nn.Module):
    def __init__(self, spectral: bool) -> None:
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm, weight_norm

        norm = spectral_norm if spectral else weight_norm
        layout = [(1, 128, 15, 1, 1, 7), (128, 128, 41, 2, 4, 20), (128, 256, 41, 2, 16, 20), (256, 512, 41, 4, 16, 20),
                  (512, 1024, 41, 4, 16, 20), (1024, 1024, 41, 1, 16, 20), (1024, 1024, 5, 1, 1, 2)]
        self.convs = nn.ModuleList(norm(nn.Conv1d(i, o, k, s, groups=g, padding=p)) for i, o, k, s, g, p in layout)
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return x.flatten(1), features


class HifiDiscriminators(nn.Module):
    """MultiPeriodDiscriminator (periods 2, 3, 5, 7, 11) and MultiScaleDiscriminator (3 scales, AvgPool1d(4, 2, 2)
    between them, spectral norm on the raw scale) of the reference, as one list of (score, feature maps)."""

    def __init__(self) -> None:
        super().__init__()
        self.periods = nn.ModuleList(HifiPeriodDiscriminator(p) for p in (2, 3, 5, 7, 11))
        self.scales = nn.ModuleList(HifiScaleDiscriminator(i == 0) for i in range(3))

    def forward(self, y: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        outputs = [d(y) for d in self.periods]
        for index, d in enumerate(self.scales):
            if index > 0:
                y = F.avg_pool1d(y, 4, 2, padding=2)
            outputs.append(d(y))
        return outputs


class HifiGanV1(nn.Module):
    """The "hifigan-v1" row: jik876/hifi-gan's V1 generator, multi-period and multi-scale discriminators, losses and
    optimizers, in AiDotNet HiFiGAN's step order. A step crops a random 32-frame mel segment and its 8192 samples from
    the 256-frame utterance, updates the discriminators on a detached generation (least squares on D(real) and
    D(fake)), then the generator on a fresh generation (least-squares adversarial + 2 x feature matching against
    no-grad real features + 45 x L1 of the full-band log-mel), each with AdamW (lr 2e-4, betas 0.8 / 0.99, weight
    decay 0.01) and no clipping. The discriminators are frozen during the generator update (AiDotNet trains only the
    generator's parameters there). One utterance per step: run at --batch-size 1. The reported loss is the
    generator's. Inference is the generator on the whole mel. Weight norm and spectral norm are torch's
    parametrizations; trainable parameters are counted over all three networks."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = HifiGenerator()
        self.discriminators = HifiDiscriminators()
        self.loss_mel = HifiMel(HIFI_RATE / 2)
        self._optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] | None = None

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.generator(mel)

    def compile_parts(self) -> None:
        """--compile: train_step calls the networks directly, so compile them rather than only forward."""
        # The protocol fixes training to 32 frames and inference to 256 at b1/b8/b16.
        # Automatic dynamic promotion can leave MPD reflection padding symbolic when the
        # frozen discriminator recompiles for generator gradients. Specialize every shape.
        self.generator = torch.compile(self.generator, dynamic=False)
        self.discriminators = torch.compile(self.discriminators, dynamic=False)

    def train_step(self, mel: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if mel.shape[0] != 1:
            raise ValueError("The hifigan-v1 row trains one utterance per step, as AiDotNet's HiFiGAN does: use "
                             "--batch-size 1.")
        if self._optimizers is None:
            settings = {"lr": 2e-4, "betas": (0.8, 0.99), "eps": 1e-8, "weight_decay": 0.01}
            self._optimizers = (torch.optim.AdamW(self.generator.parameters(), **settings),
                                torch.optim.AdamW(self.discriminators.parameters(), **settings))
        generator_optimizer, discriminator_optimizer = self._optimizers
        frames = HIFI_SEGMENT // HIFI_HOP
        start = random.randint(0, mel.shape[2] - frames)
        mel = mel[:, :, start:start + frames]
        real = audio[:, start * HIFI_HOP:(start + frames) * HIFI_HOP].unsqueeze(1)

        with torch.no_grad():
            fake = self.generator(mel)
        discriminator_loss = sum(torch.mean((1 - r) ** 2) + torch.mean(g ** 2)
                                 for (r, _), (g, _) in zip(self.discriminators(real), self.discriminators(fake)))
        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_optimizer.step()

        self.discriminators.requires_grad_(False)
        try:
            generated = self.generator(mel)
            fake_out = self.discriminators(generated)
            with torch.no_grad():
                real_out = self.discriminators(real)
                real_mel = self.loss_mel(real.squeeze(1))
            adversarial = sum(torch.mean((1 - g) ** 2) for g, _ in fake_out)
            matching = sum(torch.mean(torch.abs(r - g)) for (_, rf), (_, gf) in zip(real_out, fake_out)
                           for r, g in zip(rf, gf))
            mel_loss = F.l1_loss(self.loss_mel(generated.squeeze(1)), real_mel)
            generator_loss = adversarial + 2.0 * matching + 45.0 * mel_loss
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            generator_optimizer.step()
        finally:
            self.discriminators.requires_grad_(True)
        return generator_loss.detach()

    @staticmethod
    def make_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Three random 100-1000 Hz partials plus uniform noise, peak 0.95, and their input mel (fmax 8 kHz)."""
        samples = HIFI_FRAMES * HIFI_HOP
        t = torch.arange(samples) / HIFI_RATE
        frequency = 100 + 900 * torch.rand(batch_size, 3, 1)
        phase = 2 * math.pi * torch.rand(batch_size, 3, 1)
        audio = torch.sin(2 * math.pi * frequency * t + phase).sum(1) + 0.3 * (2 * torch.rand(batch_size, samples) - 1)
        audio = audio / audio.abs().amax(dim=1, keepdim=True) * 0.95
        with torch.no_grad():
            mel = HifiMel(8000.0)(audio)
        return mel.to(device), audio.to(device)


def make_vision_model(name: str) -> nn.Module:
    """torchvision's definition of each architecture, adjusted only where AiDotNet's builder adjusts it for a
    32x32 input, so both sides run the same network: VGG's classifier reads the 512x1x1 feature map directly
    (no 7x7 adaptive pool), and DenseNet uses the CIFAR stem (3x3 stride 1, no max pool)."""
    import torchvision.models as tvm

    if name == "resnet18":
        return tvm.resnet18(num_classes=10)
    if name == "resnet50":
        return tvm.resnet50(num_classes=10)
    if name == "vgg11":
        model = tvm.vgg11(num_classes=10)
        model.avgpool = nn.Identity()
        model.classifier = nn.Sequential(nn.Linear(512, 4096), nn.ReLU(True), nn.Linear(4096, 4096), nn.ReLU(True),
                                         nn.Linear(4096, 10))
        return model
    if name == "mobilenetv2":
        return tvm.mobilenet_v2(num_classes=10, dropout=0.0)
    if name == "densenet121":
        model = tvm.densenet121(num_classes=10)
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        return model
    if name == "vit":
        return TinyViT()
    raise ValueError(f"Unknown vision model: {name}")


VISION_MODELS = ("resnet18", "resnet50", "vgg11", "mobilenetv2", "densenet121", "vit")


def make_model(name: str) -> tuple[nn.Module, tuple[int, ...]]:
    if name == "gru":
        return GRUClassifier(), (32, 32)
    if name == "gcn":
        return GCNClassifier(), (GCN_NODES, GCN_FEATURES)
    if name == "decoder-lm":
        return DecoderLM(), (LM_POSITIONS,)
    if name == "ddpm":
        return DDPMUNet(), (DDPM_CHANNELS, DDPM_IMAGE, DDPM_IMAGE)
    if name == "whisper-tiny":
        return WhisperTiny(), (WHISPER_MELS, WHISPER_FRAMES)
    if name == "faster-rcnn":
        return FasterRcnnR50Fpn(), (3, DETECTION_IMAGE, DETECTION_IMAGE)
    if name == "efficientnet-b0":
        return EfficientNetB0(), (3, 32, 32)
    if name == "vae":
        return MnistVAE(), (VAE_PIXELS,)
    if name == "gat":
        return CoraGAT(), (GAT_NODES, GAT_FEATURES)
    if name == "dcgan":
        return DCGAN64(), (DCGAN_LATENT,)
    if name == "panns-cnn14":
        return PannsCnn14(), (1, PANNS_FRAMES, PANNS_MELS)
    if name == "wav2vec2-base":
        return Wav2Vec2BaseCtc(), (W2V_SAMPLES,)
    if name == "hifigan-v1":
        return HifiGanV1(), (HIFI_MELS, HIFI_FRAMES)
    if name in VISION_MODELS:
        return make_vision_model(name), (3, 32, 32)
    if name in TIME_SERIES_MODELS:
        factory, shape = TIME_SERIES_MODELS[name]
        return factory(), shape
    if name == "mlp":
        return MLP(), (1, 28, 28)
    if name == "cnn":
        return SmallCNN(), (1, 28, 28)
    if name == "lstm":
        return LSTMClassifier(), (32, 32)
    if name == "transformer":
        return TransformerClassifier(), (32, 32)
    # "mlp-fused" is an AiDotNet-only primitive variant; map it to the same MLP
    # so a --models list shared with the C# side does not error out here.
    if name == "mlp-fused":
        return MLP(), (1, 28, 28)
    raise ValueError(f"Unknown model: {name}")


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default) -- the same function
    as BenchStats.Quantile on the AiDotNet side, so median/IQR compare like-for-like."""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synthetic_batch(batch_size: int, shape: tuple[int, ...], device: torch.device,
                    model: nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # A row whose data is not "randn input, random class" (DDPM's clean images) generates its own batch.
    if model is not None and hasattr(model, "make_batch"):
        return model.make_batch(batch_size, device)
    if shape == (LM_POSITIONS,):
        return lm_batch(batch_size, device)
    if shape == (GCN_NODES, GCN_FEATURES):
        return gcn_batch(batch_size, device)
    x = torch.randn((batch_size, *shape), device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def benchmark_training(model: nn.Module, shape: tuple[int, ...], device: torch.device, epochs: int, batches: int, batch_size: int, breakdown: bool = False, model_name: str = "") -> TrainingResult:
    criterion = nn.CrossEntropyLoss()
    # A row whose model trains with another recipe (the detector's paper SGD) builds its own optimizer.
    # A row whose step updates several networks (the GAN) owns its optimizers and its whole step (train_step).
    optimizer = (None if hasattr(model, "train_step")
                 else model.make_optimizer() if hasattr(model, "make_optimizer")
                 else torch.optim.AdamW(model.parameters(), **getattr(model, "optimizer_settings", {"lr": 1e-3})))
    epoch_seconds: list[float] = []
    gradient_seconds: list[float] = []
    data_seconds: list[float] = []

    # Batches are generated BEFORE the timed epochs (the AiDotNet twin does the same) so epoch
    # time is pure training, not each framework's very different synthetic-data RNG cost.
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(batches):
        start_data = time.perf_counter()
        pool.append(synthetic_batch(batch_size, shape, device, model))
        synchronize(device)
        data_seconds.append(time.perf_counter() - start_data)

    start_total = time.perf_counter()
    first_loss: torch.Tensor | None = None
    # --precision bf16: the training steps under torch.autocast, as the C# side runs them under AutocastScope.
    autocast = (torch.autocast(device_type=device.type, dtype=torch.bfloat16) if PRECISION == "bf16"
                else contextlib.nullcontext())
    with ResourceMonitor(track_gpu=device.type == "cuda", gpu_id=cuda_smi_id(device), torch_device=device) as monitor, autocast:
        for epoch in range(epochs):
            # PROFILE_RANGE=1: bracket the last epoch for `nsys --capture-range=cudaProfilerApi` (same as the C# side).
            profile_epoch = os.environ.get("PROFILE_RANGE") == "1" and epoch == epochs - 1 and device.type == "cuda"
            if profile_epoch:
                synchronize(device)
                torch.cuda.profiler.start()
            mark_phase(model_name, f"training-epoch-{epoch}")
            start_epoch = time.perf_counter()
            for x, y in pool:
                if optimizer is None:
                    loss = model.train_step(x, y)
                    if first_loss is None:
                        first_loss = loss.detach()
                    continue

                optimizer.zero_grad(set_to_none=True)
                if hasattr(model, "training_loss"):
                    # A row with its own objective (DDPM's noise regression) computes its loss itself.
                    loss = model.training_loss(x, y)
                elif getattr(model, "loss_from_forward", False):
                    # A row whose forward returns its own loss given the target (DeepAR's likelihood), called through
                    # the module so a --compile run compiles the loss too.
                    loss = model(x, y)
                elif getattr(model, "regression", False):
                    # A forecaster: mean squared error of the forecast against the continuation.
                    loss = F.mse_loss(model(x), y)
                else:
                    logits = model(x)
                    # Flattened so a per-position target (the decoder LM's [B, S] next tokens) is scored like a class one.
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                if first_loss is None:
                    first_loss = loss.detach()
                # Per-step device syncs inflate the step on GPU; they run only under --breakdown,
                # which attributes time to the backward. Default timing matches AiDotNet's loop.
                if breakdown:
                    synchronize(device)
                    start_grad = time.perf_counter()
                loss.backward()
                if breakdown:
                    synchronize(device)
                    gradient_seconds.append(time.perf_counter() - start_grad)
                # Same global-norm clip AiDotNet applies by default (MaxGradNorm 1.0).
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            synchronize(device)
            epoch_seconds.append(time.perf_counter() - start_epoch)
            mark_phase(model_name, f"training-epoch-{epoch}-complete")
            if profile_epoch:
                torch.cuda.profiler.stop()
    total = time.perf_counter() - start_total
    first_loss_value = json_loss(float(first_loss.item())) if first_loss is not None else None
    last_loss_value = json_loss(float(loss.item())) if first_loss is not None else None
    # Steady state excludes epoch 0 (warmup), exactly as the AiDotNet twin does.
    steady_epochs = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    steady_sorted = sorted(steady_epochs)
    return TrainingResult(
        epoch_seconds=[round(value, 6) for value in epoch_seconds],
        total_seconds=round(total, 6),
        steady_state_epoch_seconds_avg=round(statistics.fmean(steady_epochs), 6) if steady_epochs else 0.0,
        steady_state_epoch_seconds_median=round(quantile(steady_sorted, 0.5), 6),
        steady_state_epoch_seconds_p25=round(quantile(steady_sorted, 0.25), 6),
        steady_state_epoch_seconds_p75=round(quantile(steady_sorted, 0.75), 6),
        steps_per_epoch=batches,
        gradient_seconds_avg=round(statistics.fmean(gradient_seconds), 6) if gradient_seconds else 0.0,
        data_loading_seconds_avg=round(statistics.fmean(data_seconds), 6),
        resources=monitor.summary(),
        first_loss=first_loss_value,
        last_loss=last_loss_value,
    )


def json_loss(value: float) -> float | str:
    """Encode a non-finite loss as the JSON string the C# report and compare.py use."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


@torch.inference_mode()
def benchmark_inference(model: nn.Module, shape: tuple[int, ...], device: torch.device, iterations: int, warmup: int,
                        batch_sizes: list[int], warmup_min_ms: float = 0.0, model_name: str = "") -> list[InferenceBatchResult]:
    results: list[InferenceBatchResult] = []
    for batch_size in batch_sizes:
        mark_phase(model_name, f"inference-b{batch_size}-prepare")
        x, _ = synthetic_batch(batch_size, shape, device, model)
        mark_phase(model_name, f"inference-b{batch_size}-warmup")
        warmup_times: list[float] = []
        # Same rule as the C# twin: at least warmup calls and at least warmup_min_ms of warm-up per batch size.
        warmup_start = time.perf_counter()
        while len(warmup_times) < warmup or (time.perf_counter() - warmup_start) * 1000.0 < warmup_min_ms:
            start = time.perf_counter()
            model(x)
            synchronize(device)
            warmup_times.append(time.perf_counter() - start)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        process = psutil.Process(os.getpid())
        rss_peak = process.memory_info().rss / 1024 / 1024
        mark_phase(model_name, f"inference-b{batch_size}-steady")
        steady: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(x)
            synchronize(device)
            elapsed = time.perf_counter() - start
            steady.append(elapsed)
            if os.environ.get("AIDOTNET_BENCH_EXTERNAL_MONITOR") != "1":
                rss_peak = max(rss_peak, process.memory_info().rss / 1024 / 1024)
        total = sum(steady)
        mark_phase(model_name, f"inference-b{batch_size}-complete")
        cuda_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
        # p95 latency: robust to rig-contention noise that swings the mean.
        steady_sorted = sorted(steady)
        p95_idx = min(len(steady_sorted) - 1, round(0.95 * (len(steady_sorted) - 1)))
        results.append(InferenceBatchResult(
            batch_size=batch_size,
            warmup_seconds_avg=round(statistics.fmean(warmup_times), 6),
            steady_state_latency_ms_avg=round(statistics.fmean(steady) * 1000, 3),
            steady_state_latency_ms_p95=round(steady_sorted[p95_idx] * 1000, 3),
            steady_state_latency_ms_median=round(quantile(steady_sorted, 0.5) * 1000, 3),
            steady_state_latency_ms_p25=round(quantile(steady_sorted, 0.25) * 1000, 3),
            steady_state_latency_ms_p75=round(quantile(steady_sorted, 0.75) * 1000, 3),
            throughput_samples_per_second=round((iterations * batch_size) / total, 3),
            memory_mb_peak=round(rss_peak, 3),
            gpu_tensor_allocated_mb_peak=round(cuda_peak, 3) if device.type == "cuda" else None,
        ))
    return results


_CSHARP_INT = re.compile(r"[+-]?[0-9]+")
_INT32_MAX = 2**31 - 1


def parse_inference_batches(text: str) -> list[int]:
    """Comma-separated positive batch sizes; the C# harness takes the same flag with the same contract."""
    items = [item.strip() for item in text.split(",")]
    if not items or any(not item for item in items):
        raise ValueError(f"--inference-batches must be comma-separated positive integers, got {text!r}")
    # The C# side parses with NumberStyles.Integer: optional sign, ASCII digits, Int32 range. Python's int() also takes
    # "1_0", non-ASCII digits and arbitrary size, so a value C# rejects could pass here and fail only after training.
    if any(_CSHARP_INT.fullmatch(item) is None for item in items):
        raise ValueError(f"--inference-batches must be comma-separated positive integers, got {text!r}")
    batches = [int(item) for item in items]
    if any(batch > _INT32_MAX for batch in batches):
        raise ValueError(f"--inference-batches values must fit a 32-bit integer, got {text!r}")
    if any(batch <= 0 for batch in batches):
        raise ValueError(f"--inference-batches must be positive, got {text!r}")
    return batches


def run(args: argparse.Namespace) -> dict[str, object]:
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    # Pin CPU threads for a fair head-to-head — match the AiDotNet side's
    # AIDOTNET_CPU_THREADS so the numbers reflect the kernels, not the scheduler.
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Validated before any model trains: a bad entry used to surface only after the first model's training run.
    inference_batches = parse_inference_batches(args.inference_batches)

    model_results: list[ModelResult] = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        mark_phase(name, "construction")
        model, shape = make_model(name)
        model.to(device)
        # Trainable only: the decoder LM's frozen, zeroed in_proj_bias stands in for attention AiDotNet builds without
        # a Q/K/V bias, and counting it reported 768 parameters the C# row does not have.
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if args.compile:
            # torch.compile (TorchInductor) — the "PyTorch compiled" comparison.
            # Counts params BEFORE compile (the wrapper hides .parameters()).
            # A row that trains through its own training_loss / train_step reaches it through the compiled wrapper's
            # attribute forwarding, which runs the original eager module: compile that path too, so --compile times
            # compiled training rather than only compiled inference.
            if hasattr(model, "compile_parts"):
                model.compile_parts()
            elif hasattr(model, "training_loss"):
                model.training_loss = torch.compile(model.training_loss)
            model = torch.compile(model)
        mark_phase(name, "construction-complete")
        mark_phase(name, "training-prepare")
        training = benchmark_training(model, shape, device, args.epochs, args.train_batches, args.batch_size, args.breakdown, name)
        mark_phase(name, "training-complete")
        model.eval()
        inference = benchmark_inference(model, shape, device, args.inference_iterations, args.warmup_iterations,
                                        inference_batches, args.warmup_min_ms, name)
        mark_phase(name, "complete")
        model_results.append(ModelResult(name, str(device), parameters, training, inference, MODEL_NOTES.get(name, "")))

    return {
        "framework": "PyTorch",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "protocol": {"epochs": args.epochs, "train_batches": args.train_batches, "batch_size": args.batch_size,
                     "inference_iterations": args.inference_iterations, "warmup_iterations": args.warmup_iterations,
                     "warmup_min_ms": args.warmup_min_ms, "seed": args.seed, "inference_batch_sizes": inference_batches,
                     "compile": args.compile, "precision": PRECISION},
        "results": [asdict(result) for result in model_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch MLP/CNN/LSTM/Transformer workloads (eager).")
    parser.add_argument("--models", default="mlp,cnn,lstm,transformer")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-iterations", type=int, default=100)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--warmup-min-ms", type=float, default=1000.0,
                        help="Also warm each batch size for at least this long (the C# side needs it for JIT tier-up).")
    parser.add_argument("--inference-batches", default=",".join(str(b) for b in BATCH_SIZES),
                        help="Comma-separated inference batch sizes (the C# side takes the same flag).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--breakdown", action="store_true",
                        help="sync per step to attribute time to the backward (slows GPU steps)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Pin CPU thread count (0 = PyTorch default = all cores). "
                             "Match the AiDotNet side's AIDOTNET_CPU_THREADS for a fair comparison.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap each model in torch.compile (TorchInductor) — the 'PyTorch compiled' head-to-head.")
    parser.add_argument("--output", type=Path, default=Path("../results/pytorch.json"))
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"],
                        help="Training steps under torch.autocast(bfloat16) (the C# side's AutocastScope). fp16 needs "
                             "GradScaler loss scaling in both arms and is not wired yet.")
    args = parser.parse_args()
    global PRECISION
    PRECISION = args.precision

    allocation_trace = os.environ.get("AIDOTNET_BENCH_CPU_ALLOCATION_TRACE")
    if allocation_trace:
        if os.environ.get("AIDOTNET_BENCH_DIAGNOSTICS") != "1":
            raise ValueError("Allocation profiling requires a separate diagnostic run.")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profiler:
            report = run(args)
        profiler.export_chrome_trace(allocation_trace)
    else:
        report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
