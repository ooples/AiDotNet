---
title: "RescaledCFG<T>"
description: "Rescaled Classifier-Free Guidance to prevent over-saturation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

Rescaled Classifier-Free Guidance to prevent over-saturation.

## For Beginners

When you use a very high guidance scale, images can become
oversaturated with unnaturally bright colors. This fix automatically adjusts
the brightness back to normal while keeping the guidance effect.

## How It Works

Rescales the guided noise prediction to match the standard deviation of the
conditional prediction. This prevents the color saturation and contrast
blowout that occurs at high CFG scales (e.g., > 10).

Reference: Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RescaledCFG(Double)` | Initializes a new Rescaled CFG instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

