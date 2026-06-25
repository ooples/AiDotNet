---
title: "MixedPrecisionType"
description: "Types of mixed precision training data types."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Types of mixed precision training data types.

## For Beginners

FP16 works on most GPUs, BF16 is better on newer
hardware (Ampere and later). If unsure, start with FP16.

## How It Works

Mixed precision training uses lower precision floating point numbers
to speed up training and reduce memory usage while maintaining accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `BF16` | Brain floating point (BF16) mixed precision. |
| `FP16` | Half precision (FP16) mixed precision. |
| `FP8_E4M3` | FP8 E4M3 format (4 exponent bits, 3 mantissa bits). |
| `FP8_E5M2` | FP8 E5M2 format (5 exponent bits, 2 mantissa bits). |
| `FP8_Hybrid` | Hybrid FP8 mode using E4M3 for forward pass and E5M2 for backward pass. |
| `None` | Full precision (FP32). |
| `TF32` | TensorFloat-32 (TF32) precision. |

