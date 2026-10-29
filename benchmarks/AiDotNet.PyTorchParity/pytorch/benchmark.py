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
    dotnet run ... (set AIDOTNET_BLAS_THREADS=8)

Usage:
    pip install -r requirements.txt
    python benchmark.py --models mlp,cnn,lstm,transformer --device cpu \
        --threads 8 --output ../results/pytorch.json
"""

from __future__ import annotations

import argparse
import json
import math
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


def synthetic_batch(batch_size: int, shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((batch_size, *shape), device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def benchmark_training(model: nn.Module, shape: tuple[int, ...], device: torch.device, epochs: int, batches: int, batch_size: int, breakdown: bool = False) -> TrainingResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epoch_seconds: list[float] = []
    gradient_seconds: list[float] = []
    data_seconds: list[float] = []

    # Batches are generated BEFORE the timed epochs (the AiDotNet twin does the same) so epoch
    # time is pure training, not each framework's very different synthetic-data RNG cost.
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(batches):
        start_data = time.perf_counter()
        pool.append(synthetic_batch(batch_size, shape, device))
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
                logits = model(x)
                loss = criterion(logits, y)
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
        x, _ = synthetic_batch(batch_size, shape, device)
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


def parse_inference_batches(text: str) -> list[int]:
    """Comma-separated positive batch sizes; the C# harness takes the same flag with the same contract."""
    items = [item.strip() for item in text.split(",")]
    if not items or any(not item for item in items):
        raise ValueError(f"--inference-batches must be comma-separated positive integers, got {text!r}")
    try:
        batches = [int(item) for item in items]
    except ValueError as error:
        raise ValueError(f"--inference-batches must be comma-separated positive integers, got {text!r}") from error
    if any(batch <= 0 for batch in batches):
        raise ValueError(f"--inference-batches must be positive, got {text!r}")
    return batches


def run(args: argparse.Namespace) -> dict[str, object]:
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    # Pin CPU threads for a fair head-to-head — match the AiDotNet side's
    # AIDOTNET_BLAS_THREADS so the numbers reflect the kernels, not the scheduler.
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
        parameters = sum(parameter.numel() for parameter in model.parameters())
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
                             "Match the AiDotNet side's AIDOTNET_BLAS_THREADS for a fair comparison.")
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
