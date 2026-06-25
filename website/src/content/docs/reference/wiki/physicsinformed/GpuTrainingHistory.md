---
title: "GpuTrainingHistory<T>"
description: "Training history with GPU-specific metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Training history with GPU-specific metrics.

## For Beginners

This class tracks training progress including timing and memory metrics.
These metrics help diagnose training issues and optimize performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageEpochTimeMs` | Gets or sets the average time per epoch in milliseconds. |
| `KernelTimings` | Gets or sets training timing statistics. |
| `PeakManagedMemoryBytes` | Gets or sets the peak managed (heap) memory growth during training in bytes. |
| `TotalTrainingTimeMs` | Gets or sets the total training time in milliseconds. |
| `UseGpuAcceleration` | Gets or sets whether GPU acceleration was used during training. |

