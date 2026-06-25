---
title: "QuantizationGranularity"
description: "Specifies the granularity level for quantization scaling factors."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the granularity level for quantization scaling factors.
Finer granularity preserves more accuracy but requires more storage for scale/zero-point values.

## For Beginners

When compressing numbers, we need "scaling factors" to convert
between big and small numbers. Granularity determines how many different scaling factors we use:

## How It Works

**Analogy:** Think of it like setting brightness on a photo:

**Memory Overhead:** Finer granularity requires storing more scaling factors:

**Research Reference:** K-Quant in llama.cpp uses a two-level scheme (PerBlock with super-blocks)
achieving excellent quality with minimal overhead.

## Fields

| Field | Summary |
|:-----|:--------|
| `PerBlock` | Per-block quantization with super-blocks (K-Quant style from llama.cpp). |
| `PerChannel` | Per-channel quantization - separate scale and zero-point for each output channel. |
| `PerGroup` | Per-group quantization - separate scale and zero-point for each group of N consecutive elements. |
| `PerRow` | Row-wise quantization - separate scale per row of a weight matrix. |
| `PerTensor` | Per-tensor quantization - single scale and zero-point for the entire tensor. |

