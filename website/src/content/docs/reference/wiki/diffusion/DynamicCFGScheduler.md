---
title: "DynamicCFGScheduler<T>"
description: "Dynamic Classifier-Free Guidance that adjusts scale per timestep."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

Dynamic Classifier-Free Guidance that adjusts scale per timestep.

## For Beginners

Instead of using the same guidance strength for every step,
this uses stronger guidance at the start (to get the big picture right) and
lighter guidance at the end (to keep fine details natural).

## How It Works

Applies a time-dependent guidance scale that starts high for structural coherence
in early (noisy) steps and decreases toward the end for fine detail preservation.
This reduces over-saturation artifacts common with high static CFG scales.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DynamicCFGScheduler(Double,Double)` | Initializes a new Dynamic CFG scheduler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

