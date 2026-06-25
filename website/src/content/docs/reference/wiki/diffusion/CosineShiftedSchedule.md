---
title: "CosineShiftedSchedule<T>"
description: "Cosine-shifted noise schedule for resolution-adapted diffusion training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers.NoiseSchedules`

Cosine-shifted noise schedule for resolution-adapted diffusion training.

## For Beginners

At higher resolutions, the same amount of noise is less disruptive
because there are more pixels to average over. This schedule compensates by adding more
noise at higher resolutions, so the model trains consistently regardless of image size.

## How It Works

Shifts the cosine noise schedule based on image resolution to account for the fact
that higher-resolution images need more noise to fully corrupt. The shift factor
scales with resolution, ensuring consistent effective noise levels across resolutions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineShiftedSchedule(Double)` | Initializes a new instance with the specified shift factor. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAlphasCumprod(Int32)` | Computes shifted cosine alpha cumulative products. |

