---
title: "Float8E5M2"
description: "Represents an 8-bit floating point number in E5M2 format (5 exponent bits, 2 mantissa bits)."
section: "API Reference"
---

`Structs` · `AiDotNet.MixedPrecision`

Represents an 8-bit floating point number in E5M2 format (5 exponent bits, 2 mantissa bits).

## For Beginners

E5M2 is an 8-bit format with a larger range but less precision than E4M3.
It's designed for gradients during backpropagation, which can have a wider range of values.

## How It Works

**Format Details:**

- 1 sign bit
- 5 exponent bits (bias = 15)
- 2 mantissa bits
- Range: ±57344
- Smallest positive (subnormal): ~0.0000152588 (2^-16)

**Use Cases:** Best for gradients in neural network backward passes where the larger
dynamic range helps prevent gradient underflow/overflow.

**Best Practice:** Use E5M2 for gradients with dynamic loss scaling to handle the
reduced precision while leveraging the larger dynamic range.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInfinity` | Returns true if this value is infinity (positive or negative). |
| `IsNaN` | Returns true if this value is NaN. |
| `IsNegative` | Returns true if this value is negative. |
| `IsZero` | Returns true if this value is zero. |
| `RawValue` | Gets the raw byte value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareTo(Float8E5M2)` |  |
| `Equals(Float8E5M2)` |  |
| `Equals(Object)` |  |
| `FromFloat(Single)` | Creates a Float8E5M2 from a single-precision float. |
| `GetHashCode` |  |
| `ToFloat` | Converts this Float8E5M2 to a single-precision float. |
| `ToString` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MaxValue` | Maximum representable value in E5M2 format (57344). |
| `MinPositive` | Minimum representable positive value in E5M2 format (smallest subnormal: 2^-16). |
| `NaN` | Represents NaN (Not a Number). |
| `NegativeInfinity` | Represents negative infinity. |
| `One` | Represents positive one. |
| `PositiveInfinity` | Represents positive infinity. |
| `Zero` | Represents positive zero. |

