"""PyTorch side of the AiDotNet ⇄ PyTorch parity benchmark.

Twin of the C# harness (../Program.cs): same four reference models
(MLP / CNN / LSTM / Transformer) with matching layer shapes, the same
training + multi-batch inference measurement loop, and the same JSON schema,
so the two reports compare directly via compare.py.

IMPORTANT (fair comparison): PyTorch here runs EAGER (no torch.compile). The
AiDotNet side runs its own compiled/fused path; pitting it against a separately
torch.compile'd PyTorch would compare two different compilation stacks rather
than the kernels. Pin both sides to the same CPU thread count:
    python benchmark.py --threads 8 ...
    dotnet run ... (set AIDOTNET_CPU_THREADS=8)

Usage:
    pip install -r requirements.txt
    python benchmark.py --models mlp,cnn,lstm,transformer --device cpu \
        --threads 8 --output ../results/pytorch.json
"""

from __future__ import annotations

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

BATCH_SIZES = [1, 8, 32, 128]


@dataclass
class ResourceSummary:
    cpu_percent_avg: float
    rss_mb_peak: float
    gpu_util_percent_avg: float | None
    gpu_mem_mb_peak: float | None


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
    def __init__(self, sample_seconds: float = 0.1, track_gpu: bool = True) -> None:
        self.sample_seconds = sample_seconds
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
        self.process.cpu_percent(interval=None)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.cpu.append(self.process.cpu_percent(interval=None))
            self.rss_mb.append(self.process.memory_info().rss / 1024 / 1024)
            if self.track_gpu:
                gpu = query_nvidia_smi()
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
        )


def query_nvidia_smi() -> tuple[float, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
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


# nn.GRU applies the reset gate after the recurrent product (n = tanh(W_in x + b_in + r * (W_hn h + b_hn))); AiDotNet's
# GRULayer applies it before (n = tanh(W_in x + b_in + W_hn (r * h) + b_hn), Cho et al. / Keras reset_after=False).
GRU_NOTE = ("Different recurrence: nn.GRU applies the reset gate after the recurrent product, AiDotNet's GRULayer "
            "before it (Keras reset_after=False). Same sizes and parameter shapes; a timing comparison, not parity.")
MODEL_NOTES = {"gru": GRU_NOTE}


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
    if name in VISION_MODELS:
        return make_vision_model(name), (3, 32, 32)
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


def benchmark_training(model: nn.Module, shape: tuple[int, ...], device: torch.device, epochs: int, batches: int, batch_size: int, breakdown: bool = False) -> TrainingResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), **getattr(model, "optimizer_settings", {"lr": 1e-3}))
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
    with ResourceMonitor(track_gpu=device.type == "cuda") as monitor:
        for epoch in range(epochs):
            # PROFILE_RANGE=1: bracket the last epoch for `nsys --capture-range=cudaProfilerApi` (same as the C# side).
            profile_epoch = os.environ.get("PROFILE_RANGE") == "1" and epoch == epochs - 1 and device.type == "cuda"
            if profile_epoch:
                synchronize(device)
                torch.cuda.profiler.start()
            start_epoch = time.perf_counter()
            for x, y in pool:

                optimizer.zero_grad(set_to_none=True)
                if hasattr(model, "training_loss"):
                    # A row with its own objective (DDPM's noise regression) computes its loss itself.
                    loss = model.training_loss(x, y)
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
                        batch_sizes: list[int], warmup_min_ms: float = 0.0) -> list[InferenceBatchResult]:
    results: list[InferenceBatchResult] = []
    for batch_size in batch_sizes:
        x, _ = synthetic_batch(batch_size, shape, device, model)
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
        steady: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(x)
            synchronize(device)
            elapsed = time.perf_counter() - start
            steady.append(elapsed)
            rss_peak = max(rss_peak, process.memory_info().rss / 1024 / 1024)
        total = sum(steady)
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
            memory_mb_peak=round(max(rss_peak, cuda_peak), 3),
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
        model, shape = make_model(name)
        model.to(device)
        # Trainable only: the decoder LM's frozen, zeroed in_proj_bias stands in for attention AiDotNet builds without
        # a Q/K/V bias, and counting it reported 768 parameters the C# row does not have.
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if args.compile:
            # torch.compile (TorchInductor) — the "PyTorch compiled" comparison.
            # Counts params BEFORE compile (the wrapper hides .parameters()).
            model = torch.compile(model)
        training = benchmark_training(model, shape, device, args.epochs, args.train_batches, args.batch_size, args.breakdown)
        model.eval()
        inference = benchmark_inference(model, shape, device, args.inference_iterations, args.warmup_iterations,
                                        inference_batches, args.warmup_min_ms)
        model_results.append(ModelResult(name, str(device), parameters, training, inference, MODEL_NOTES.get(name, "")))

    return {
        "framework": "PyTorch",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
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
    args = parser.parse_args()

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
