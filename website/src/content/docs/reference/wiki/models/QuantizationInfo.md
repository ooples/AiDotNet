---
title: "QuantizationInfo"
description: "Contains information about model quantization applied during or after training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains information about model quantization applied during or after training.
Provides metrics on compression ratio, accuracy impact, and quantization parameters.

## For Beginners

After quantizing (compressing) your model, this class tells you
how much smaller it got, what technique was used, and other useful information about the
compression process.

## How It Works

**Key Metrics:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ActivationBitWidth` | Gets the bit width used for activation quantization (if applicable). |
| `ActivationsQuantized` | Gets whether activations were also quantized (in addition to weights). |
| `BitWidth` | Gets the bit width used for quantized weights. |
| `CalibrationMethod` | Gets the calibration method used to determine quantization parameters. |
| `CalibrationSamples` | Gets the number of calibration samples used for quantization. |
| `CompressionRatio` | Gets the compression ratio (original size / quantized size). |
| `Granularity` | Gets the quantization granularity (PerTensor, PerChannel, PerGroup). |
| `GroupSize` | Gets the group size used for per-group quantization. |
| `IsQuantized` | Gets whether quantization was applied to this model. |
| `IsSymmetric` | Gets whether symmetric quantization was used. |
| `LayerInfo` | Gets per-layer quantization information if available. |
| `Mode` | Gets the quantization mode used (Int8, Float16, etc.). |
| `None` | Creates a default QuantizationInfo indicating no quantization was applied. |
| `OriginalSizeBytes` | Gets the original model size in bytes before quantization. |
| `QATMethod` | Gets the QAT method used if QAT was enabled. |
| `QuantizationTimeMs` | Gets the time taken to perform quantization in milliseconds. |
| `QuantizedParameters` | Gets the number of parameters that were actually quantized. |
| `QuantizedPercentage` | Gets the percentage of parameters that were quantized. |
| `QuantizedSizeBytes` | Gets the quantized model size in bytes after quantization. |
| `Strategy` | Gets the quantization strategy (algorithm) used (GPTQ, AWQ, etc.). |
| `TotalParameters` | Gets the total number of quantized parameters. |
| `UsedQAT` | Gets whether Quantization-Aware Training (QAT) was used. |
| `Warnings` | Gets any warnings or notes generated during quantization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary of the quantization. |

