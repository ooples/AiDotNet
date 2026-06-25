---
title: "InitializationStrategyBase<T>"
description: "Base class for initialization strategies providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Initialization`

Base class for initialization strategies providing common functionality.

## For Beginners

This base class contains the shared code that all initialization
strategies need, avoiding duplication and ensuring consistent behavior across different
initialization methods.

## How It Works

This abstract base class provides shared implementation for all initialization strategies,
including common helper methods for weight initialization patterns like Xavier/Glorot
and He initialization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InitializationStrategyBase` | Initializes a new instance of the `InitializationStrategyBase` class using the framework's default thread-safe non-deterministic RNG. |
| `InitializationStrategyBase(Random)` | Initializes a new instance of the `InitializationStrategyBase` class using the supplied `Random` (for reproducible / seeded initialization). |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` |  |
| `LoadFromExternal` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateUnlockedSeededRandom(Int32)` | Returns a non-thread-safe `Random` for SINGLE-THREADED chunk init paths. |
| `FillChunkDouble(Span<Double>,Double,Double,Random)` | Sequential Box-Muller fill of a span — inner helper used by both the sequential fast path and the parallel chunk workers. |
| `HeNormalInitialize(Tensor<>,Int32)` | Initializes weights using He/Kaiming normal initialization. |
| `HeUniformInitialize(Tensor<>,Int32)` | Initializes weights using He/Kaiming uniform initialization. |
| `InitializeBiases(Tensor<>)` |  |
| `InitializeWeights(Tensor<>,Int32,Int32)` |  |
| `SampleGaussian(Double,Double)` | Samples a value from a Gaussian (normal) distribution using the Box-Muller transform. |
| `UniformFillDouble(Double[],Int32,Int32,Double)` | Parallel uniform fill: fills `dst`[offset..offset+length] with samples from U(-limit, limit). |
| `UniformFillFloat(Single[],Int32,Int32,Double)` | Float variant of `Double)`. |
| `WithSeededRandom(Random)` | Returns an instance of this strategy that samples from the supplied (typically seeded) `Random`, preserving the strategy's distribution. |
| `XavierFillDouble(Double[],Int32,Int32,Double,Double)` | Fills a span with `N(0, stddev)` samples clipped to ±`clipBound`, using a paired Box-Muller transform that produces two samples per pair of uniform draws — halves the `Double)`/`Double)` call count vs. |
| `XavierFillFloat(Single[],Int32,Int32,Double,Double)` | Float variant of `Double)`. |
| `XavierNormalInitialize(Tensor<>,Int32,Int32)` | Initializes weights using Xavier/Glorot normal initialization. |
| `XavierUniformInitialize(Tensor<>,Int32,Int32)` | Initializes weights using Xavier/Glorot uniform initialization. |
| `ZeroInitializeBiases(Tensor<>)` | Initializes biases to zero (common default). |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | The numeric operations helper for type T. |
| `Random` | Random number generator used for sampling weights. |

