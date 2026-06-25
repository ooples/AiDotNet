---
title: "QuantizationConfiguration"
description: "Configuration for model quantization - comprehensive settings for PTQ and QAT."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Optimization.Quantization`

Configuration for model quantization - comprehensive settings for PTQ and QAT.

## For Beginners

Quantization compresses your model by using smaller numbers.
This configuration lets you control exactly how that compression happens.

## How It Works

**Quick Start Examples:**

**Research References:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AWQProtectionPercentage` | Gets or sets the protection percentage for AWQ strategy. |
| `AWQScaleSearchOptions` | Gets or sets the scale search options for AWQ grid search optimization. |
| `ActivationBitWidth` | Gets or sets the bit width for activation quantization (if QuantizeActivations is true). |
| `BitWidth` | Gets the bit width for the current quantization mode. |
| `CalibrationMethod` | Gets or sets the calibration method. |
| `CategoryBitWidths` | Gets or sets per-category bit-width overrides for mixed-precision quantization. |
| `CustomLayerParams` | Gets or sets custom quantization parameters per layer (by layer name). |
| `DefaultBitWidth` | Gets the default bit width for the current quantization mode. |
| `EffectiveBitWidth` | Gets the effective bit width (target or default based on mode). |
| `GPTQActOrder` | Gets or sets whether to use ActOrder optimization in GPTQ. |
| `GPTQDampingFactor` | Gets or sets the damping factor for GPTQ Hessian computation. |
| `Granularity` | Gets or sets the quantization granularity (where to apply scaling factors). |
| `GroupSize` | Gets or sets the group size for per-group quantization. |
| `HistogramPercentile` | Gets or sets the percentile to use for histogram-based calibration. |
| `MaxScaleFactor` | Gets or sets the maximum scale factor to prevent overflow. |
| `MinScaleFactor` | Gets or sets the minimum scale factor to prevent underflow. |
| `Mode` | Gets or sets the quantization mode (Int8, Float16, etc.). |
| `NumCalibrationSamples` | Gets or sets the number of calibration samples to use. |
| `QATMethod` | Gets or sets the QAT method to use when UseQuantizationAwareTraining is true. |
| `QATWarmupEpochs` | Gets or sets the number of QAT warmup epochs before enabling fake quantization. |
| `QuantizeActivations` | Gets or sets whether to quantize only weights or both weights and activations. |
| `SkipLayers` | Gets or sets layers to skip during quantization. |
| `SmoothQuantAlpha` | Gets or sets the smoothing factor alpha for SmoothQuant strategy. |
| `Strategy` | Gets or sets the quantization strategy (algorithm) to use. |
| `TargetBitWidth` | Gets or sets the target bit width for weight quantization. |
| `UsePerChannelQuantization` | Gets or sets whether to use per-channel quantization. |
| `UseQuantizationAwareTraining` | Gets or sets whether to use quantization-aware training (QAT). |
| `UseSymmetricQuantization` | Gets or sets whether to use symmetric quantization (default: true). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForAWQ(Int32,Double)` | Creates a configuration for AWQ 4-bit quantization. |
| `ForDynamic` | Creates a configuration for dynamic quantization (weights only). |
| `ForFloat16` | Creates a configuration for FP16 quantization. |
| `ForGPTQ(Int32,Boolean)` | Creates a configuration for GPTQ 4-bit quantization. |
| `ForInt8(CalibrationMethod)` | Creates a configuration for INT8 quantization. |
| `ForMixedPrecision(Int32,Int32,Int32)` | Creates a mixed-precision configuration for layer-aware quantization. |
| `ForQAT(Int32,QATMethod)` | Creates a configuration for Quantization-Aware Training (QAT). |
| `ForQLoRA` | Creates a configuration optimized for 4-bit QLoRA fine-tuning. |
| `ForSmoothQuant(Double)` | Creates a configuration for SmoothQuant W8A8 quantization. |
| `GetBitWidthForCategory(LayerCategory)` | Gets the effective bit-width for a specific layer category, considering per-category overrides in `CategoryBitWidths`. |
| `GetBitWidthForLayer(LayerInfo<>)` | Gets the effective bit-width for a specific layer, checking name-based overrides first, then category-based overrides, then the default. |
| `GetFullPrecisionBitWidth` | Returns the full-precision bit width for the current quantization mode. |

