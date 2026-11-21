"""Large-data mode for benchmark.py (--data DIR): the PyTorch twin of ScaleBench.cs.

A model trains for whole epochs over a dataset written by scale/prepare_data.py and scores the held-out split after
every epoch. Both frameworks read the same memory-mapped files, visit the samples in the same shuffled order
(permutation(), SplitMix64 Fisher-Yates), normalise a byte b of channel c to (b / 255 - mean[c]) / std[c], drop the last
partial batch, and read the training loss every LOSS_EVERY steps.

PyTorch uses its fastest input path measured on the benchmark machine. The training loop gathers each batch's uint8
samples itself, into pinned memory on CUDA, and copies them asynchronously, so the gather overlaps the GPU's work on the
previous step. The model normalises the bytes itself (Normalized), so under torch.compile the conversion is part of the
compiled graph and only uint8 data is copied to the device. A background prefetch thread (as AiDotNet uses) and
DataLoader worker processes were both slower for PyTorch: see the large-data section of the README.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOSS_EVERY = 100
TEST_BATCH = 250
# Untimed updates before epoch 0, on its first batches, so compilation and autotuning stay out of the epoch times.
WARMUP_STEPS = 20
MASK64 = (1 << 64) - 1


def permutation(count: int, seed: int, epoch: int) -> np.ndarray:
    """ScaleShuffle.Permutation in ScaleBench.cs, step for step."""
    order = list(range(count))
    state = (seed * 1_000_003 + epoch) & MASK64
    for i in range(count - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        j = z % (i + 1)
        order[i], order[j] = order[j], order[i]
    return np.asarray(order, dtype=np.int64)


class ScaleDataset:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.name: str = meta["name"]
        self.sample_shape: list[int] = meta["sample_shape"]
        self.classes: int = meta["classes"]
        self.train_samples: int = meta["train_samples"]
        self.test_samples: int = meta["test_samples"]
        self.mean: list[float] = meta["mean"]
        self.std: list[float] = meta["std"]
        self.train_sha256: str = meta["files"]["train-x"]["sha256"]
        self.sample_size = int(np.prod(self.sample_shape))
        if self.sample_size % len(self.mean) or len(self.mean) != len(self.std):
            raise ValueError(f"{self.name}: {len(self.mean)} channel statistics do not divide a {self.sample_size}-byte sample")
        self.train_x = self._map(directory / "train-x.u8", self.train_samples)
        self.test_x = self._map(directory / "test-x.u8", self.test_samples)
        self.train_y = self._labels(directory / "train-y.u8", self.train_samples)
        self.test_y = self._labels(directory / "test-y.u8", self.test_samples)

    def _map(self, path: Path, count: int) -> np.ndarray:
        expected = count * self.sample_size
        if path.stat().st_size != expected:
            raise ValueError(f"{path} has {path.stat().st_size} bytes, expected {expected}")
        return np.memmap(path, dtype=np.uint8, mode="r", shape=(count, self.sample_size))

    @staticmethod
    def _labels(path: Path, count: int) -> np.ndarray:
        labels = np.fromfile(path, dtype=np.uint8)
        if labels.shape[0] != count:
            raise ValueError(f"{path} has {labels.shape[0]} labels, expected {count}")
        return labels.astype(np.int64)



class Normalized(nn.Module):
    """Wraps a classifier so it reads a dataset's uint8 samples [B, sample] and normalises them channel by channel, in
    ScaleBench's operation order: (b / 255 - mean[c]) / std[c]. mean and std are buffers, not parameters."""

    def __init__(self, inner: nn.Module, data: ScaleDataset, model_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.inner = inner
        self.model_shape = tuple(model_shape)
        channels = len(data.mean)
        self.register_buffer("mean", torch.tensor(data.mean, dtype=torch.float32).view(1, channels, 1))
        self.register_buffer("std", torch.tensor(data.std, dtype=torch.float32).view(1, channels, 1))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = (raw.view(raw.shape[0], self.mean.shape[1], -1).float() / 255.0 - self.mean) / self.std
        return self.inner(x.view(raw.shape[0], *self.model_shape))


def train(model: nn.Module, model_shape: tuple[int, ...], data: ScaleDataset, device: torch.device, epochs: int,
          batch_size: int, seed: int, optimizer: torch.optim.Optimizer, cuda_graphs: bool, synchronize, begin_step,
          monitor_factory) -> dict[str, object]:
    if int(np.prod(model_shape)) != data.sample_size:
        raise ValueError(f"{data.name} samples hold {data.sample_size} values; the model reads {int(np.prod(model_shape))}")
    if data.test_samples % TEST_BATCH:
        raise ValueError(f"{data.name} has {data.test_samples} test samples, not a multiple of {TEST_BATCH}")
    criterion = nn.CrossEntropyLoss()
    pin = device.type == "cuda"
    steps = data.train_samples // batch_size
    epochs_report: list[dict[str, object]] = []
    first_loss: float | None = None
    cumulative = 0.0
    with monitor_factory() as monitor:
        warmup_order = permutation(data.train_samples, seed, 0)
        model.train()
        for step in range(min(WARMUP_STEPS, steps)):
            index = warmup_order[step * batch_size:(step + 1) * batch_size]
            x = torch.from_numpy(data.train_x[index]).to(device)
            y = torch.from_numpy(data.train_y[index]).to(device)
            begin_step(cuda_graphs)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            begin_step(cuda_graphs)
            model(torch.from_numpy(np.array(data.test_x[:TEST_BATCH])).to(device))
        model.train()
        synchronize(device)
        for epoch in range(epochs):
            order = permutation(data.train_samples, seed, epoch)
            losses: list[float] = []
            waited = 0.0
            model.train()
            start = time.perf_counter()
            for step in range(steps):
                gather_start = time.perf_counter()
                index = order[step * batch_size:(step + 1) * batch_size]
                raw_x = torch.from_numpy(data.train_x[index])
                raw_y = torch.from_numpy(data.train_y[index])
                if pin:
                    raw_x, raw_y = raw_x.pin_memory(), raw_y.pin_memory()
                waited += time.perf_counter() - gather_start
                x = raw_x.to(device, non_blocking=pin)
                y = raw_y.to(device, non_blocking=pin)
                begin_step(cuda_graphs)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                # Same global-norm clip AiDotNet applies by default (MaxGradNorm 1.0).
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % LOSS_EVERY == 0:
                    value = float(loss.item())
                    first_loss = value if first_loss is None else first_loss
                    losses.append(value)
            synchronize(device)
            train_seconds = time.perf_counter() - start
            cumulative += train_seconds

            eval_start = time.perf_counter()
            accuracy = evaluate(model, data, device, cuda_graphs, begin_step)
            synchronize(device)
            eval_seconds = time.perf_counter() - eval_start
            report = dict(epoch=epoch, train_seconds=round(train_seconds, 6),
                          samples_per_second=round(steps * batch_size / train_seconds, 6),
                          data_wait_seconds=round(waited, 6),
                          train_loss=round(sum(losses) / len(losses), 6) if losses else None,
                          eval_seconds=round(eval_seconds, 6), test_accuracy=round(accuracy, 6),
                          cumulative_train_seconds=round(cumulative, 6))
            epochs_report.append(report)
            print(f"[scale] {data.name} epoch {epoch}: {train_seconds:.1f}s, {report['samples_per_second']:.0f} samples/s, "
                  f"loss {report['train_loss']:.4f}, test accuracy {accuracy:.2%}", flush=True)
    return dict(dataset=data.name, train_sha256=data.train_sha256, train_samples=data.train_samples,
                test_samples=data.test_samples, batch_size=batch_size, test_batch_size=TEST_BATCH, steps_per_epoch=steps,
                warmup_steps=min(WARMUP_STEPS, steps),
                loss_every=LOSS_EVERY, first_loss=first_loss, epochs=epochs_report, total_train_seconds=round(cumulative, 6),
                resources=monitor.summary())


def evaluate(model: nn.Module, data: ScaleDataset, device: torch.device, cuda_graphs: bool, begin_step) -> float:
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device=device)
    with torch.no_grad():
        for start in range(0, data.test_samples, TEST_BATCH):
            x = torch.from_numpy(np.array(data.test_x[start:start + TEST_BATCH])).to(device)
            y = torch.from_numpy(data.test_y[start:start + TEST_BATCH]).to(device)
            begin_step(cuda_graphs)
            correct += (model(x).argmax(dim=1) == y).sum()
    model.train()
    return int(correct.item()) / data.test_samples
