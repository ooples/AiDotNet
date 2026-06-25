---
title: "MXFP4Quantizer<T, TInput, TOutput>"
description: "MXFP4 (Microscaling FP4) quantizer - uses shared exponents for efficient 4-bit floating point."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Formats`

MXFP4 (Microscaling FP4) quantizer - uses shared exponents for efficient 4-bit floating point.
Part of the OCP (Open Compute Project) Microscaling specification.

## For Beginners

MXFP4 is a 4-bit floating point format where groups of numbers
share a common scale (exponent). This allows better representation of values across
different magnitudes while staying compact.

## How It Works

**How It Works:**

**Format Details:**

**Key Features:**

**Reference:** OCP Microscaling Formats Specification (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MXFP4Quantizer(QuantizationConfiguration,Int32)` | Initializes a new instance of the MXFP4Quantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `BlockSize` | Gets the block size used for shared scaling. |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `ComputeBlockScales(Vector<>)` | Computes shared scales for each block of parameters. |
| `DecodeFromMXFP4(Int32)` | Decodes a 4-bit MXFP4 representation to its value. |
| `EncodeToMXFP4(Double)` | Encodes a value to its 4-bit MXFP4 representation. |
| `FindNearestMXFP4Value(Double)` | Finds the nearest MXFP4 representation for a value. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeWithMXFP4(Vector<>,Int32)` | Quantizes parameters using MXFP4 format with microscaling. |

