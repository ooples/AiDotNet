---
title: "AudioEffectBase<T>"
description: "Base class for audio effects processors."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Audio.Effects`

Base class for audio effects processors.

## For Beginners

for provides AI safety functionality. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioEffectBase(Int32,Double)` | Initializes a new AudioEffectBase. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Bypass` |  |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `LatencySamples` |  |
| `Mix` |  |
| `Name` |  |
| `Parameters` |  |
| `SampleRate` |  |
| `TailSamples` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddParameter(String,String,,,,String,String)` | Adds a parameter to the effect. |
| `DbToLinear(Double)` | Converts decibels to linear amplitude. |
| `GetParameter(String)` |  |
| `LinearToDb(Double)` | Converts linear amplitude to decibels. |
| `OnParameterChanged(String,)` | Called when a parameter value changes. |
| `Process(Tensor<>)` |  |
| `ProcessInPlace(Span<>)` |  |
| `ProcessSample()` |  |
| `ProcessSampleInternal()` | Processes a single sample through the effect. |
| `Reset` |  |
| `SetParameter(String,)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `_parameters` | Mutable parameters dictionary. |

