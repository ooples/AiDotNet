---
title: "EqBand<T>"
description: "Represents a single EQ band with biquad filter."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Represents a single EQ band with biquad filter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EqBand(INumericOperations<>,Int32,Double,Double,Double,EqFilterType)` | Creates a new EQ band. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FilterType` | Filter type. |
| `Frequency` | Center/corner frequency in Hz. |
| `GainDb` | Gain in dB. |
| `Q` | Q factor (bandwidth). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Process()` | Processes a sample through the biquad filter. |
| `Reset` | Resets the filter state. |
| `SetParameters(Double,Double,Double)` | Sets the band parameters and recalculates coefficients. |

