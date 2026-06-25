---
title: "FlashAttentionPrecision"
description: "Precision modes for Flash Attention computation."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks.Attention`

Precision modes for Flash Attention computation.

## Fields

| Field | Summary |
|:-----|:--------|
| `Float16` | Use 16-bit floating point (half precision). |
| `Float32` | Use 32-bit floating point (single precision). |
| `Mixed` | Use mixed precision (FP16 for matmul, FP32 for softmax). |

