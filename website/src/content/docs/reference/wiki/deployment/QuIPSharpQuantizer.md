---
title: "QuIPSharpQuantizer<T, TInput, TOutput>"
description: "QuIP# (Quantization with Incoherence Processing Sharp) quantizer for extreme 2-bit quantization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization.Strategies`

QuIP# (Quantization with Incoherence Processing Sharp) quantizer for extreme 2-bit quantization.
Uses Hadamard transforms for incoherence and lattice-based codebooks for optimal quantization.

## For Beginners

QuIP# achieves incredibly aggressive 2-bit quantization
(just 4 possible values per weight!) while maintaining reasonable accuracy. It uses
mathematical transformations to spread information more evenly before quantization.

## How It Works

**How It Works:**

**Key Features:**

**Reference:** Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard
Incoherence and Lattice Codebooks" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QuIPSharpQuantizer(QuantizationConfiguration,Int32)` | Initializes a new instance of the QuIPSharpQuantizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` |  |
| `IsCalibrated` | Gets whether the quantizer has been calibrated. |
| `Mode` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyHadamard(Double[])` | Applies the Walsh-Hadamard transform (multiplication-free). |
| `Calibrate(IFullModel<,,>,IEnumerable<>)` |  |
| `ComputeScaleFactors(Vector<>)` | Computes scale factors from parameters. |
| `FindNearestCodebookValue(Double)` | Finds the nearest codebook value for 2-bit quantization. |
| `GetScaleFactor(String)` |  |
| `GetZeroPoint(String)` |  |
| `NextPowerOfTwo(Int32)` | Returns the next power of two greater than or equal to n. |
| `Quantize(IFullModel<,,>,QuantizationConfiguration)` |  |
| `QuantizeWithQuIPSharp(Vector<>)` | Applies QuIP# quantization to parameters. |

