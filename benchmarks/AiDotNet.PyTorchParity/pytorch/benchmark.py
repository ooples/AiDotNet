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
    gradient_seconds_avg: float
    data_loading_seconds_avg: float
    resources: ResourceSummary


@dataclass
class InferenceBatchResult:
    batch_size: int
    warmup_seconds_avg: float
    steady_state_latency_ms_avg: float
    steady_state_latency_ms_p95: float
    throughput_samples_per_second: float
    memory_mb_peak: float


@dataclass
class ModelResult:
    model: str
    device: str
    parameters: int
    training: TrainingResult
    inference: list[InferenceBatchResult]


class ResourceMonitor:
    def __init__(self, sample_seconds: float = 0.1) -> None:
        self.sample_seconds = sample_seconds
        self.process = psutil.Process(os.getpid())
        self.cpu: list[float] = []
        self.rss_mb: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "ResourceMonitor":
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.proj = nn.Linear(32, 64)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.proj(x))
        return self.head(encoded.mean(dim=1))


def make_model(name: str) -> tuple[nn.Module, tuple[int, ...]]:
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synthetic_batch(batch_size: int, shape: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((batch_size, *shape), device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def benchmark_training(model: nn.Module, shape: tuple[int, ...], device: torch.device, epochs: int, batches: int, batch_size: int) -> TrainingResult:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epoch_seconds: list[float] = []
    gradient_seconds: list[float] = []
    data_seconds: list[float] = []

    start_total = time.perf_counter()
    with ResourceMonitor() as monitor:
        for _ in range(epochs):
            start_epoch = time.perf_counter()
            for _ in range(batches):
                start_data = time.perf_counter()
                x, y = synthetic_batch(batch_size, shape, device)
                synchronize(device)
                data_seconds.append(time.perf_counter() - start_data)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                synchronize(device)
                start_grad = time.perf_counter()
                loss.backward()
                synchronize(device)
                gradient_seconds.append(time.perf_counter() - start_grad)
                optimizer.step()
            synchronize(device)
            epoch_seconds.append(time.perf_counter() - start_epoch)
    total = time.perf_counter() - start_total
    return TrainingResult(
        epoch_seconds=[round(value, 6) for value in epoch_seconds],
        total_seconds=round(total, 6),
        gradient_seconds_avg=round(statistics.fmean(gradient_seconds), 6),
        data_loading_seconds_avg=round(statistics.fmean(data_seconds), 6),
        resources=monitor.summary(),
    )


@torch.inference_mode()
def benchmark_inference(model: nn.Module, shape: tuple[int, ...], device: torch.device, iterations: int, warmup: int) -> list[InferenceBatchResult]:
    results: list[InferenceBatchResult] = []
    for batch_size in BATCH_SIZES:
        x, _ = synthetic_batch(batch_size, shape, device)
        warmup_times: list[float] = []
        for _ in range(warmup):
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
        p95_idx = min(len(steady_sorted) - 1, int(round(0.95 * (len(steady_sorted) - 1))))
        results.append(InferenceBatchResult(
            batch_size=batch_size,
            warmup_seconds_avg=round(statistics.fmean(warmup_times), 6),
            steady_state_latency_ms_avg=round(statistics.fmean(steady) * 1000, 3),
            steady_state_latency_ms_p95=round(steady_sorted[p95_idx] * 1000, 3),
            throughput_samples_per_second=round((iterations * batch_size) / total, 3),
            memory_mb_peak=round(max(rss_peak, cuda_peak), 3),
        ))
    return results


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

    model_results: list[ModelResult] = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        model, shape = make_model(name)
        model.to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        training = benchmark_training(model, shape, device, args.epochs, args.train_batches, args.batch_size)
        model.eval()
        inference = benchmark_inference(model, shape, device, args.inference_iterations, args.warmup_iterations)
        model_results.append(ModelResult(name, str(device), parameters, training, inference))

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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0,
                        help="Pin CPU thread count (0 = PyTorch default = all cores). "
                             "Match the AiDotNet side's AIDOTNET_BLAS_THREADS for a fair comparison.")
    parser.add_argument("--output", type=Path, default=Path("../results/pytorch.json"))
    args = parser.parse_args()

    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
