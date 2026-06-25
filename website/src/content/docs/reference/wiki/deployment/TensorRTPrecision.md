---
title: "TensorRTPrecision"
description: "TensorRT precision modes for inference."
section: "API Reference"
---

`Enums` · `AiDotNet.Deployment.TensorRT`

TensorRT precision modes for inference.

## For Beginners

Precision affects speed vs accuracy trade-off:

- FP32 (32-bit float): Full precision, most accurate, slowest
- FP16 (16-bit float): Half precision, good balance, ~2x faster than FP32
- INT8 (8-bit integer): Quantized, fastest, requires calibration data

For most use cases, FP16 provides a good balance. Use INT8 only when
you have calibration data and need maximum throughput.

## Fields

| Field | Summary |
|:-----|:--------|
| `FP16` | 16-bit floating point (half precision, ~2x faster) |
| `FP32` | 32-bit floating point (full precision) |
| `INT8` | 8-bit integer (quantized, fastest, requires calibration) |

