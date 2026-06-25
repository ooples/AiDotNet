---
title: "IGpuAcceleratedPINN<T>"
description: "Interface for Physics-Informed Neural Networks that support GPU acceleration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Interface for Physics-Informed Neural Networks that support GPU acceleration.

## How It Works

For Beginners:
This interface marks PINNs that can take advantage of GPU acceleration during training.
GPU acceleration can provide significant speedups for:

- Batch processing of collocation points
- Parallel derivative computations
- Matrix operations in the forward and backward passes

Implementing this interface signals that the PINN can use GPU resources when available.

## Properties

| Property | Summary |
|:-----|:--------|
| `GpuConfig` | Gets or sets the GPU acceleration configuration. |
| `IsGpuAvailable` | Gets a value indicating whether GPU is currently available and ready for use. |
| `UseGpuAcceleration` | Gets or sets whether GPU acceleration is enabled for training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeGpu(GpuAccelerationConfig)` | Initializes GPU resources for accelerated training. |
| `ReleaseGpuResources` | Releases GPU resources. |

