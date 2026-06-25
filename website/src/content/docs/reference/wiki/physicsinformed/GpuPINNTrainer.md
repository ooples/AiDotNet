---
title: "GpuPINNTrainer<T>"
description: "Provides GPU-accelerated training for Physics-Informed Neural Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Provides GPU-accelerated training for Physics-Informed Neural Networks.

## How It Works

For Beginners:
This trainer provides GPU-accelerated training methods for PINNs.
It can significantly speed up training by:

1. Processing large batches of collocation points in parallel
2. Performing matrix operations on the GPU
3. Using asynchronous data transfers to overlap computation

Usage:
```cs
var trainer = new GpuPINNTrainer<double>(myPinn);
var history = trainer.Train(epochs: 10000, options: GpuPINNTrainingOptions.Default);
```

The trainer automatically falls back to CPU if GPU is not available.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GpuPINNTrainer(PhysicsInformedNeuralNetwork<>,GpuPINNTrainingOptions)` | Initializes a new instance of the GPU PINN trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsUsingGpu` | Gets whether GPU is currently being used for training. |
| `Network` | Gets the underlying PINN being trained. |
| `Options` | Gets the current training options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateLoss(Tensor<>,Tensor<>)` | Performs a forward pass and computes loss (does not update weights). |
| `GetGpuMemoryInfo` | Gets GPU memory usage statistics. |
| `ReleaseGpuResources` | Releases GPU resources. |
| `Train(Tensor<>,Tensor<>,Int32,Double,Boolean)` | Trains the PINN with GPU acceleration. |
| `TryInitializeGpu` | Attempts to initialize GPU resources. |
| `UpdateOptions(GpuPINNTrainingOptions)` | Updates training options. |

