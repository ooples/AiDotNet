---
title: "PINNGpuMemoryInfo"
description: "GPU memory usage information for PINN training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

GPU memory usage information for PINN training.

## How It Works

**Note:** A value of -1 for any property indicates that the information
is not available through the current GPU backend.

For accurate memory profiling, use external tools such as:

- NVIDIA: nvidia-smi, nvml library
- AMD: rocm-smi
- General: Visual Studio GPU profiler

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableMemoryBytes` | Gets or sets the available GPU memory in bytes. |
| `IsMemoryInfoAvailable` | Gets whether memory information is available. |
| `TotalMemoryBytes` | Gets or sets the total GPU memory in bytes. |
| `UsagePercentage` | Gets the usage percentage, or -1 if memory information is not available. |
| `UsedMemoryBytes` | Gets or sets the used GPU memory in bytes. |

