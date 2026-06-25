---
title: "Float8E4M3"
description: "Represents an 8-bit floating point number in E4M3 format (4 exponent bits, 3 mantissa bits)."
section: "API Reference"
---

`Structs` · `AiDotNet.MixedPrecision`

Represents an 8-bit floating point number in E4M3 format (4 exponent bits, 3 mantissa bits).

## For Beginners

E4M3 is like a compressed version of regular floating point numbers.
It uses only 8 bits (1 byte) instead of 32 bits, making it 4x smaller. The trade-off is
reduced precision and range.

## How It Works

**Format Details:**

- 1 sign bit
- 4 exponent bits (bias = 7)
- 3 mantissa bits
- Range: ±448
- Smallest positive: ~0.001953125

**Use Cases:** Best for weights and activations in neural network forward passes
where values are typically well-bounded.

**Hardware:** NVIDIA H100/H200 GPUs have native FP8 Tensor Cores that can process
E4M3 values at 2x the throughput of FP16.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsNaN` | Returns true if this value is NaN. |
| `IsNegative` | Returns true if this value is negative. |
| `IsZero` | Returns true if this value is zero. |
| `RawValue` | Gets the raw byte value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareTo(Float8E4M3)` |  |
| `Equals(Float8E4M3)` |  |
| `Equals(Object)` |  |
| `FromFloat(Single)` | Creates a Float8E4M3 from a single-precision float. |
| `GetHashCode` |  |
| `ToFloat` | Converts this Float8E4M3 to a single-precision float. |
| `ToString` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MaxValue` | Maximum representable value in E4M3 format (448). |
| `MinPositive` | Minimum representable positive value in E4M3 format. |
| `NaN` | Represents NaN (Not a Number). |
| `One` | Represents positive one. |
| `Zero` | Represents positive zero. |

