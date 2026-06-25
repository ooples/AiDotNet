---
title: "MixedPrecisionConfig"
description: "Configuration settings for mixed-precision training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MixedPrecision`

Configuration settings for mixed-precision training.

## For Beginners

This class contains all the settings you can adjust for mixed-precision training.
The default values work well for most models, but you can customize them based on your specific needs.

Key concepts:

- **Loss Scaling**: Prevents small gradients from becoming zero in FP16
- **Dynamic Scaling**: Automatically adjusts the loss scale during training
- **Master Weights**: FP32 copy of parameters for precise updates
- **Working Weights**: FP16 copy used for forward/backward passes
- **FP8 (New)**: 8-bit formats for 2x throughput on H100+ GPUs

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedPrecisionConfig` | Creates a configuration with default recommended settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableDynamicScaling` | Enable dynamic loss scaling (default: true). |
| `FP8BackwardFormat` | Format to use for backward pass in FP8 mode (default: E5M2). |
| `FP8ExcludedLayers` | Layers to keep in higher precision (FP16/BF16) even when using FP8. |
| `FP8ForwardFormat` | Format to use for forward pass in FP8 mode (default: E4M3). |
| `FP8PerTensorScaling` | Whether to use per-tensor scaling for FP8 (default: true). |
| `Fp32BatchNorm` | Whether to keep batch normalization layers in FP32 (default: true). |
| `Fp32GradientAccumulation` | Whether to accumulate gradients in FP32 (default: true). |
| `Fp32Loss` | Whether to keep loss computation in FP32 (default: true). |
| `InitialLossScale` | Initial loss scale factor (default: 65536 = 2^16). |
| `MaxLossScale` | Maximum allowed loss scale (default: 16777216 = 2^24). |
| `MinLossScale` | Minimum allowed loss scale (default: 1.0). |
| `PrecisionType` | The precision type to use for mixed-precision training (default: FP16). |
| `ScaleBackoffFactor` | Factor by which to multiply scale when decreasing (default: 0.5). |
| `ScaleGrowthFactor` | Factor by which to multiply scale when increasing (default: 2.0). |
| `ScaleGrowthInterval` | Number of consecutive successful updates before increasing scale (default: 2000). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggressive` | Creates an aggressive configuration optimized for maximum throughput. |
| `Conservative` | Creates a conservative configuration optimized for training stability. |
| `ForBF16` | Creates a configuration for BF16 precision (Ampere+ GPUs). |
| `ForFP8` | Creates a configuration for FP8 hybrid mode (H100+ GPUs). |
| `ForFP8Transformers` | Creates a configuration for FP8 training on transformer models. |
| `ToString` | Gets a summary of the configuration. |

