---
title: "AdaptiveProjectedGuidance<T>"
description: "Adaptive Projected Guidance (APG) for diffusion model inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Guidance`

Adaptive Projected Guidance (APG) for diffusion model inference.

## For Beginners

Standard guidance can sometimes push the image in bad
directions, causing artifacts. APG is smarter — it only keeps the "useful"
part of the guidance while discarding the part that causes problems.

## How It Works

APG projects the guidance direction to reduce components that cause artifacts.
It decomposes the guidance vector into parallel and perpendicular components
relative to the conditional prediction, keeping only the beneficial part.

Reference: Ahn et al., "Adaptive Projected Guidance", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveProjectedGuidance(Double)` | Initializes a new Adaptive Projected Guidance instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` |  |

