---
title: "AugmentationContext<T>"
description: "Provides runtime context for augmentation operations including random state, training mode, and spatial targets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Provides runtime context for augmentation operations including random state,
training mode, and spatial targets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AugmentationContext(Boolean,Nullable<Int32>)` | Creates a new augmentation context. |
| `AugmentationContext(Random,Boolean)` | Creates a new augmentation context with a provided random instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchIndex` | Gets the batch index (if applicable). |
| `IsTraining` | Gets whether the context is in training mode. |
| `Metadata` | Gets additional metadata for the current augmentation. |
| `Random` | Gets the random number generator for this context. |
| `SampleIndex` | Gets the sample index within the batch (if applicable). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateChildContext(Int32)` | Creates a child context with the same random state but different indices. |
| `GetRandomBool` | Gets a random boolean with 50% probability. |
| `GetRandomDouble(Double,Double)` | Gets a random value within the specified range. |
| `GetRandomInt(Int32,Int32)` | Gets a random integer within the specified range. |
| `SampleBeta(Double,Double)` | Samples from a Beta distribution (used by Mixup/CutMix). |
| `SampleGamma(Double)` | Samples from a Gamma distribution. |
| `SampleGaussian(Double,Double)` | Samples from a Gaussian (normal) distribution. |
| `SampleStandardNormal` | Samples from a standard normal distribution. |
| `ShouldApply(Double)` | Determines whether an augmentation with the given probability should be applied. |

