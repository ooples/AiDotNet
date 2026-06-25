---
title: "QuantizationType<T>"
description: "Specifies the type of 4-bit quantization to use for base layer weights."
section: "API Reference"
---

`Enums` · `AiDotNet.LoRA.Adapters`

Specifies the type of 4-bit quantization to use for base layer weights.

## How It Works

Same quantization types as QLoRA. The alternating optimization works with both.

## Fields

| Field | Summary |
|:-----|:--------|
| `INT4` | 4-bit integer quantization with uniform spacing (-8 to 7). |
| `NF4` | 4-bit Normal Float quantization optimized for normally distributed weights. |

