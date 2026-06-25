---
title: "QuantizationStrategy"
description: "Specifies the quantization strategy (algorithm) to use for model compression."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the quantization strategy (algorithm) to use for model compression.
Different strategies offer varying trade-offs between accuracy, speed, and compression ratio.

## For Beginners

Think of quantization strategies like different compression algorithms
for photos - some preserve more detail, some compress more aggressively. Each strategy uses
different math to decide how to convert big numbers (32-bit) to small numbers (8-bit or 4-bit).

## How It Works

**Strategy Comparison:**

**Research References:**

## Fields

| Field | Summary |
|:-----|:--------|
| `AWQ` | AWQ (Activation-aware Weight Quantization) - protects important weights based on activation magnitudes. |
| `Dynamic` | Dynamic quantization - computes scale/zero-point at runtime based on actual values. |
| `GPTQ` | GPTQ (Generative Pre-trained Transformer Quantization) - uses second-order Hessian information to minimize quantization error. |
| `MinMax` | Simple MinMax quantization - uses minimum and maximum values to determine scale. |
| `QuIPSharp` | QuIP# (Quantization with Incoherence Processing) - extreme 2-bit quantization using Hadamard transforms and lattice codebooks. |
| `SmoothQuant` | SmoothQuant - transfers quantization difficulty from activations to weights using mathematical smoothing. |
| `SpinQuant` | SpinQuant - uses learned rotation matrices to reduce outliers before quantization. |

