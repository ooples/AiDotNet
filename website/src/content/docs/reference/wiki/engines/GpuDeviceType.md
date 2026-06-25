---
title: "GpuDeviceType"
description: "GPU device type for acceleration."
section: "API Reference"
---

`Enums` · `AiDotNet.Engines`

GPU device type for acceleration.

## For Beginners

Different GPU types work with different graphics cards:

- **Auto**: Automatically select best available (CUDA → OpenCL → HIP → CPU)
- **CUDA**: NVIDIA GPUs only (GeForce, RTX, Quadro, Tesla, A100, H100)
- **OpenCL**: Cross-platform (AMD, Intel, NVIDIA, Apple)
- **CPU**: Force CPU-only execution (no GPU)

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select best available GPU (CUDA → OpenCL → HIP → CPU). |
| `CPU` | Force CPU-only execution (no GPU acceleration). |
| `CUDA` | NVIDIA CUDA (GeForce, RTX, Quadro, Tesla, A100, H100). |
| `OpenCL` | OpenCL (AMD, Intel, NVIDIA, Apple - cross-platform). |

