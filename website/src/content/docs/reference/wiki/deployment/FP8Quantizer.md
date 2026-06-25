---
title: "FP8Quantizer<T, TInput, TOutput>"
description: "FP8 (8-bit Floating Point) quantizer supporting E4M3 and E5M2 formats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Formats`

FP8 (8-bit Floating Point) quantizer supporting E4M3 and E5M2 formats.
Provides better outlier handling than INT8 while maintaining 8-bit efficiency.

## For Beginners

FP8 is a newer 8-bit format that uses floating-point representation
instead of integers. It's better at handling outliers (extreme values) and requires less
calibration than INT8.

## How It Works

**FP8 Formats:**

**Key Features:**

**Reference:** NVIDIA FP8 specification and various hardware vendor implementations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FP8Quantizer(FP8Format,QuantizationConfiguration)` | Initializes a new instance of the FP8Quantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `Format` | Gets the FP8 format being used (E4M3 or E5M2). |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ByteToE4M3(Byte)` | Converts a byte to E4M3 double value. |
| `ByteToE5M2(Byte)` | Converts a byte to E5M2 double value. |
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `E4M3ToByte(Double)` | Converts an E4M3 double value to byte. |
| `E5M2ToByte(Double)` | Converts an E5M2 double value to byte. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeToFP8(Vector<>,QuantizationConfiguration)` | Quantizes parameters to FP8 format. |
| `ToE4M3(Double)` | Converts to E4M3 format (4 exponent bits, 3 mantissa bits). |
| `ToE5M2(Double)` | Converts to E5M2 format (5 exponent bits, 2 mantissa bits). |
| `ToFP8(Double)` | Converts a double to FP8 representation and back (simulating FP8 precision loss). |

