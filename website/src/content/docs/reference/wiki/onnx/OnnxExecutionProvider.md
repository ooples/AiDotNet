---
title: "OnnxExecutionProvider"
description: "Specifies the execution provider (hardware accelerator) for ONNX model inference."
section: "API Reference"
---

`Enums` · `AiDotNet.Onnx`

Specifies the execution provider (hardware accelerator) for ONNX model inference.

## For Beginners

Think of execution providers as different engines:

- **CPU**: Works everywhere, slowest but most compatible
- **CUDA**: NVIDIA GPUs, much faster than CPU
- **TensorRT**: NVIDIA GPUs with extra optimizations, fastest for NVIDIA
- **DirectML**: Windows GPUs (AMD, Intel, NVIDIA), good cross-vendor support
- **CoreML**: Apple Silicon (M1/M2/M3), fastest on Mac

## How It Works

Execution providers allow ONNX models to run on different hardware accelerators.
The order of fallback is typically: CUDA/TensorRT → DirectML → CPU.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically select the best available provider. |
| `CoreML` | Apple CoreML execution provider for Apple Silicon. |
| `Cpu` | CPU execution provider (default, always available). |
| `Cuda` | NVIDIA CUDA execution provider for GPU acceleration. |
| `DirectML` | DirectML execution provider for Windows GPU acceleration. |
| `NNAPI` | NNAPI execution provider for Android devices. |
| `OpenVINO` | OpenVINO execution provider for Intel hardware. |
| `ROCm` | ROCm execution provider for AMD GPUs. |
| `TensorRT` | NVIDIA TensorRT execution provider for optimized GPU inference. |

