---
title: "DataType"
description: "DataType — Enums in AiDotNet.Onnx.Protobuf."
section: "API Reference"
---

`Enums` · `AiDotNet.Onnx.Protobuf`

_No summary documentation available yet._

## Fields

| Field | Summary |
|:-----|:--------|
| `Bfloat16` | Non-IEEE floating-point format based on IEEE754 single-precision floating-point number truncated to 16 bits. |
| `Bool` | bool |
| `Complex128` | complex with float64 real and imaginary components |
| `Complex64` | complex with float32 real and imaginary components |
| `Float` | Basic types. |
| `Float16` | IEEE754 half-precision floating-point format (16 bits wide). |
| `Float8E4M3Fn` | Non-IEEE floating-point format based on papers FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433, 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf. |
| `Float8E4M3Fnuz` | float 8, mostly used for coefficients, supports nan, not inf, no negative zero |
| `Float8E5M2` | follows IEEE 754, supports nan, inf, mostly used for gradients |
| `Float8E5M2Fnuz` | follows IEEE 754, supports nan, not inf, mostly used for gradients, no negative zero |
| `Int16` | int16_t |
| `Int32` | int32_t |
| `Int4` | Signed integer in range [-8, 7], using two's-complement representation |
| `Int64` | int64_t |
| `Int8` | int8_t |
| `String` | string |
| `Uint16` | uint16_t |
| `Uint4` | 4-bit data-types |
| `Uint8` | uint8_t |

