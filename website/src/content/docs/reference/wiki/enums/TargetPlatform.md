---
title: "TargetPlatform"
description: "Target hardware platforms for model deployment and optimization."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Target hardware platforms for model deployment and optimization.

## How It Works

**For Beginners:** Different devices and platforms have different hardware capabilities.
This enum helps you specify where your AI model will run, allowing the library to optimize
the model specifically for that platform. For example:

- CPU: Traditional computer processors (slowest but most compatible)
- GPU: Graphics cards (much faster for AI workloads)
- TensorRT: NVIDIA's optimized AI inference engine (fastest for NVIDIA GPUs)
- Mobile: Smartphones and tablets (limited power, needs optimization)
- Edge: Small devices like Raspberry Pi or Arduino (very limited resources)

## Fields

| Field | Summary |
|:-----|:--------|
| `CPU` | Generic CPU - most compatible but slower for AI workloads |
| `CoreML` | iOS with CoreML - Apple's machine learning framework |
| `Edge` | Edge devices (Raspberry Pi, etc.) - very limited resources |
| `GPU` | Generic GPU - faster than CPU for AI computations |
| `Mobile` | Mobile devices (iOS/Android) - requires size and power optimizations |
| `NNAPI` | Android with NNAPI - Android's Neural Networks API |
| `TFLite` | TensorFlow Lite - lightweight models for mobile and edge devices |
| `TensorRT` | NVIDIA GPU with TensorRT - optimized for NVIDIA GPUs |
| `WebAssembly` | WebAssembly - run AI models in web browsers |

