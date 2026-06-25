---
title: "BetaSchedule"
description: "Defines the types of beta (noise variance) schedules available for diffusion models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the types of beta (noise variance) schedules available for diffusion models.

## For Beginners

Think of this like choosing how to gradually add static to a TV signal.

- Linear: Add static evenly - each step adds about the same amount
- ScaledLinear: Start slow, then add more - common in image generation (Stable Diffusion)
- SquaredCosine: Smooth S-curve - often produces better quality results

The choice affects both training efficiency and generation quality.

## How It Works

The beta schedule controls how noise variance changes across timesteps during the
diffusion process. Different schedules have different characteristics and are suited
for different applications.

## Fields

| Field | Summary |
|:-----|:--------|
| `Linear` | Linear interpolation between beta start and end values. |
| `ScaledLinear` | Scaled linear schedule commonly used in latent diffusion models. |
| `SquaredCosine` | Squared cosine schedule for improved diffusion models. |

