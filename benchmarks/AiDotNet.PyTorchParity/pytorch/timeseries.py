"""PyTorch side of the time-series forecaster rows (#2561).

Reads the series that TimeSeriesBench.cs wrote (series.csv), trains a twin of each AiDotNet forecaster with the same
architecture, loss, optimizer, window and batch, and writes pytorch_timeseries_<device>[_compiled].json beside it.

Every row gives PyTorch its best setup: torch.compile on the model (with --compile), the fused Adam kernel on CUDA,
the training windows gathered on the device from one resident series tensor, and the epoch loss accumulated on the
device and read back once per epoch rather than once per step.

Usage:
    python timeseries.py DIR --device cuda --compile [--epochs 6] [--seed 42] [--threads 8] [--rows nbeats]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def quantile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default), the same as the C# side's Quantile."""
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


class NBeatsBlock(nn.Module):
    """One AiDotNet NBEATSBlock: hidden_layers ReLU layers (the first from the lookback), then a backcast theta and a
    forecast theta head, each expanded through a fixed polynomial basis (t / length) ** p, p = 0..degree."""

    def __init__(self, lookback: int, horizon: int, hidden: int, hidden_layers: int, degree: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = lookback
        for _ in range(hidden_layers):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.fc = nn.Sequential(*layers)
        self.theta_backcast = nn.Linear(hidden, degree + 1)
        self.theta_forecast = nn.Linear(hidden, degree + 1)
        powers = torch.arange(degree + 1, dtype=torch.float32)
        self.register_buffer("basis_backcast", (torch.arange(lookback, dtype=torch.float32) / lookback)[:, None] ** powers)
        self.register_buffer("basis_forecast", (torch.arange(horizon, dtype=torch.float32) / horizon)[:, None] ** powers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        return self.theta_backcast(h) @ self.basis_backcast.T, self.theta_forecast(h) @ self.basis_forecast.T


class NBeats(nn.Module):
    """The doubly-residual stack (Oreshkin et al. 2020 section 3.2), as NBEATSModel.RunForwardStack runs it."""

    def __init__(self, blocks: int, lookback: int, horizon: int, hidden: int, hidden_layers: int, degree: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(NBeatsBlock(lookback, horizon, hidden, hidden_layers, degree) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = None
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast
            forecast = block_forecast if forecast is None else forecast + block_forecast
        return forecast


def train_windowed(name: str, model: nn.Module, series: np.ndarray, lookback: int, horizon: int, batch_size: int,
                   epochs: int, lr: float, clip: float, seed: int, device: torch.device, compile_model: bool,
                   config: dict) -> dict:
    """Trains a lookback -> horizon forecaster for `epochs` epochs over every full window of the normalized series,
    shuffled each epoch, one Adam step per batch. Matches NBEATSModel.TrainCore's loop."""
    torch.manual_seed(seed)
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    step_model = torch.compile(model) if compile_model else model

    y = torch.tensor(series, dtype=torch.float32, device=device)
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-10)
    # Window starts idx with idx >= lookback and idx + horizon <= len, as NBEATSModel filters them.
    starts = torch.arange(lookback, len(series) - horizon + 1, device=device)
    lookback_offsets = torch.arange(-lookback, 0, device=device)
    horizon_offsets = torch.arange(0, horizon, device=device)

    use_fused = device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, fused=use_fused)
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device=device).manual_seed(seed)
    steps_per_epoch = math.ceil(len(starts) / batch_size)

    epoch_seconds: list[float] = []
    losses: list[float] = []
    print(f"[timeseries] {name}: training {epochs} epoch(s) on {device.type}, {steps_per_epoch} step(s) per epoch, "
          f"{parameters} params, compile={compile_model}")
    total_start = time.perf_counter()
    for _ in range(epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        order = starts[torch.randperm(len(starts), device=device, generator=generator)]
        loss_sum = torch.zeros((), device=device)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            inputs = y[idx[:, None] + lookback_offsets]
            targets = y[idx[:, None] + horizon_offsets]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(step_model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            loss_sum += loss.detach()
        epoch_loss = (loss_sum / steps_per_epoch).item()  # one host read per epoch; also the epoch's barrier
        epoch_seconds.append(time.perf_counter() - start)
        losses.append(epoch_loss)
    total = time.perf_counter() - total_start

    steady = epoch_seconds[1:]
    print(f"[timeseries] {name}: epochs {', '.join(f'{s:.3f}' for s in epoch_seconds)} s; "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return {
        "model": name,
        "parameters": parameters,
        "config": config,
        "steps_per_epoch": steps_per_epoch,
        "epoch_seconds": epoch_seconds,
        "total_seconds": total,
        "steady_state_epoch_seconds_median": quantile(steady, 0.5),
        "steady_state_epoch_seconds_p25": quantile(steady, 0.25),
        "steady_state_epoch_seconds_p75": quantile(steady, 0.75),
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def nbeats_row(series: np.ndarray, args: argparse.Namespace, device: torch.device) -> dict:
    """NBEATSModelOptions defaults: 30 stacks x 1 block, 4 hidden layers of 512, degree 2, lookback 10, horizon 5,
    batch 1024, Adam lr 1e-3, global-norm clip 1.0, MSE."""
    config = {"stacks": 30, "hidden": 512, "hidden_layers": 4, "polynomial_degree": 2, "lookback": 10, "horizon": 5,
              "batch_size": 1024}
    model = NBeats(config["stacks"], config["lookback"], config["horizon"], config["hidden"], config["hidden_layers"],
                   config["polynomial_degree"])
    return train_windowed("nbeats", model, series, config["lookback"], config["horizon"], config["batch_size"],
                          args.epochs, 1e-3, 1.0, args.seed, device, args.compile, config)


ROWS = {"nbeats": nbeats_row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compile", action="store_true", help="torch.compile each model (the comparison bar)")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; match AIDOTNET_CPU_THREADS (0 = all)")
    parser.add_argument("--rows", default="", help="comma-separated row names; empty runs every row")
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("--epochs must be at least 2: epoch 0 is warm-up and compilation")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    series = np.loadtxt(args.directory / "series.csv", delimiter=",", skiprows=1, dtype=np.float64)
    only = [r for r in args.rows.split(",") if r]
    rows = [ROWS[name](series, args, device) for name in ROWS if not only or name in only]
    report = {
        "framework": f"PyTorch {torch.__version__}",
        "device": device.type,
        "compiled": args.compile,
        "cpu_threads": torch.get_num_threads(),
        "series_length": len(series),
        "epochs": args.epochs,
        "rows": rows,
    }
    suffix = "_compiled" if args.compile else ""
    path = args.directory / f"pytorch_timeseries_{device.type}{suffix}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[timeseries] report written to {path}")


if __name__ == "__main__":
    main()
