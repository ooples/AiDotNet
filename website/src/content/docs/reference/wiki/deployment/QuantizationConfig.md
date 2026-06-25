---
title: "QuantizationConfig"
description: "Configuration for model quantization - compressing models by using lower precision numbers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for model quantization - compressing models by using lower precision numbers.

## For Beginners

Quantization makes your AI model smaller and faster by using smaller numbers.
Think of it like compressing a high-quality photo - it takes less space but might lose a little quality.

## How It Works

**Why use quantization?**

**Quick Start Examples:**

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationMethod` | Gets or sets the calibration method used to determine optimal scaling factors (default: MinMax). |
| `CalibrationSamples` | Gets or sets the number of calibration samples to use (default: 100). |
| `Granularity` | Gets or sets the quantization granularity (where to apply scaling factors). |
| `GroupSize` | Gets or sets the group size for per-group quantization (default: 128). |
| `Mode` | Gets or sets the quantization mode (default: None). |
| `QATMethod` | Gets or sets the QAT method to use when UseQuantizationAwareTraining is true. |
| `QATWarmupEpochs` | Gets or sets the number of warmup epochs before enabling quantization in QAT. |
| `QuantizeActivations` | Gets or sets whether to quantize only weights or both weights and activations (default: false). |
| `Strategy` | Gets or sets the quantization strategy (algorithm) to use. |
| `TargetBitWidth` | Gets or sets the target bit width for weight quantization. |
| `UseQuantizationAwareTraining` | Gets or sets whether to use Quantization-Aware Training (QAT) instead of Post-Training Quantization. |
| `UseSymmetricQuantization` | Gets or sets whether to use symmetric quantization (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAWQ(Int32)` | Creates a configuration for AWQ 4-bit quantization. |
| `ForGPTQ(Int32)` | Creates a configuration for GPTQ 4-bit quantization. |
| `ForQAT(Int32,QATMethod)` | Creates a configuration for Quantization-Aware Training (QAT). |
| `ForSmoothQuant` | Creates a configuration for SmoothQuant W8A8 quantization. |
| `ToQuantizationConfiguration` | Converts this config to a QuantizationConfiguration for internal use. |

