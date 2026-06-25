---
title: "GpuPINNTrainingOptions"
description: "Configuration options for GPU-accelerated PINN training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.PhysicsInformed`

Configuration options for GPU-accelerated PINN training.

## How It Works

For Beginners:
These options control how GPU acceleration is used during physics-informed training.
GPU acceleration can significantly speed up training by parallelizing operations
across thousands of collocation points simultaneously.

Key settings:

- EnableGpu: Master switch for GPU acceleration
- BatchSizeGpu: Larger batches benefit more from GPU parallelism
- ParallelDerivativeComputation: Compute derivatives for multiple points at once
- AsyncTransfers: Overlap CPU/GPU data transfers with computation

## Properties

| Property | Summary |
|:-----|:--------|
| `AsyncTransfers` | Gets or sets whether to use asynchronous GPU transfers. |
| `BatchSizeGpu` | Gets or sets the batch size for GPU operations. |
| `CpuOnly` | Creates options that disable GPU entirely (CPU-only mode). |
| `Default` | Creates default options suitable for most GPUs. |
| `EnableGpu` | Gets or sets whether to enable GPU acceleration (default: true if GPU available). |
| `GpuConfig` | Gets or sets the GPU acceleration configuration. |
| `HighEnd` | Creates options optimized for high-end GPUs (RTX 4090, A100, H100). |
| `LowMemory` | Creates options optimized for memory-constrained GPUs. |
| `MinPointsForGpu` | Gets or sets the minimum number of collocation points to trigger GPU usage. |
| `NumStreams` | Gets or sets the number of CUDA streams for parallel operations. |
| `ParallelDerivativeComputation` | Gets or sets whether to compute derivatives in parallel across the batch. |
| `UseMixedPrecision` | Gets or sets whether to use mixed precision (FP16) for forward/backward passes. |
| `UsePinnedMemory` | Gets or sets whether to pin memory for faster GPU transfers. |
| `VerboseLogging` | Gets or sets whether to enable verbose GPU logging for debugging. |

