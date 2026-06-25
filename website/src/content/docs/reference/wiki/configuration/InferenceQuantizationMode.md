---
title: "InferenceQuantizationMode"
description: "Specifies the weight quantization mode for inference optimization."
section: "API Reference"
---

`Enums` · `AiDotNet.Configuration`

Specifies the weight quantization mode for inference optimization.

## For Beginners

These modes control how model weights are compressed for faster inference.

Each mode offers a different trade-off between compression and accuracy:

- INT8: 4x compression, excellent accuracy (per-row scaling)
- FP8: 4x compression, better outlier handling (floating-point format)
- NF4: 8x compression, optimized for normally-distributed weights (QLoRA format)

## Fields

| Field | Summary |
|:-----|:--------|
| `None` | No weight quantization. |
| `WeightOnlyFP8` | Per-row FP8 E4M3 weight-only quantization (4x compression). |
| `WeightOnlyInt8` | Per-row INT8 weight-only quantization (4x compression). |
| `WeightOnlyNF4` | Per-group NF4 (4-bit NormalFloat) weight-only quantization (8x compression). |

