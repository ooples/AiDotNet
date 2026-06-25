---
title: "QuantizationMode"
description: "Specifies the quantization mode for model optimization and export."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the quantization mode for model optimization and export.

## Fields

| Field | Summary |
|:-----|:--------|
| `Dynamic` | Dynamic quantization (quantize at runtime) |
| `Float16` | 16-bit floating point quantization |
| `Float32` | 32-bit floating point (full precision, no quantization) |
| `Int8` | 8-bit integer quantization |
| `Mixed` | Mixed precision (some layers quantized, some not) |
| `None` | No quantization applied |

