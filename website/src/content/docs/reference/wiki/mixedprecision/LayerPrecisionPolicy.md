---
title: "LayerPrecisionPolicy"
description: "Defines precision policies for different layer types during mixed-precision training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MixedPrecision`

Defines precision policies for different layer types during mixed-precision training.

## For Beginners

Not all layers behave well in low precision. This class helps you
configure which layers should stay in higher precision (FP32 or FP16) even when using
mixed-precision training.

## How It Works

**Why Some Layers Need Higher Precision:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerPrecisionPolicy(MixedPrecisionType)` | Creates a new layer precision policy with the specified default precision. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultPrecision` | Gets the default precision for layers not matching any rule. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddPattern(String,MixedPrecisionType)` | Adds a pattern-based precision rule. |
| `ForBF16` | Creates the default policy for BF16 mixed-precision training. |
| `ForFP16` | Creates the default policy for FP16 mixed-precision training. |
| `ForFP8` | Creates the default policy for FP8 mixed-precision training. |
| `ForFP8ConvNets` | Creates a policy for convolutional networks with FP8. |
| `ForFP8Transformers` | Creates a policy for transformer models with FP8. |
| `FullPrecision` | Creates a fully FP32 policy (no mixed precision). |
| `GetExcludedPatterns(MixedPrecisionType)` | Gets all layer patterns that should be excluded from a given precision level. |
| `GetPrecision(String)` | Gets the precision to use for a layer with the given name. |
| `KeepInFP16(String)` | Keeps layers matching the pattern in FP16 (useful when default is FP8). |
| `KeepInFP32(String)` | Keeps layers matching the pattern in full precision (FP32). |
| `SetPrecision(String,MixedPrecisionType)` | Sets the precision for a specific layer by exact name. |
| `ShouldExcludeForPrecision(MixedPrecisionType,MixedPrecisionType)` | Determines if a layer's required precision should exclude it from a target precision. |
| `ShouldSkipMixedPrecision(String)` | Determines if a layer should skip mixed precision entirely (stay in FP32). |
| `ShouldUseHigherPrecision(String)` | Determines if a layer should be kept in higher precision (FP32 or FP16). |
| `ToString` |  |

