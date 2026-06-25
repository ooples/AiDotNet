---
title: "StrengthBasedScheduling<T>"
description: "Strength-based scheduling for img2img and inpainting denoising control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Schedulers`

Strength-based scheduling for img2img and inpainting denoising control.

## For Beginners

In image editing with diffusion models, "strength" controls how
much the AI changes the original image. Low strength (0.2) makes small tweaks, high
strength (0.9) almost completely regenerates. This class handles the math of converting
that simple 0-1 slider into the right technical settings for the diffusion process.

## How It Works

Controls how much of the original image is preserved vs. regenerated in img2img and
inpainting pipelines. Maps a user-facing "strength" parameter (0.0-1.0) to the
appropriate starting timestep in the noise schedule, determining how many denoising
steps to run and at what noise level to begin.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StrengthBasedScheduling(Int32,Double)` | Initializes strength-based scheduling. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultStrength` | Gets the default denoising strength. |
| `TotalTimesteps` | Gets the total number of timesteps in the full schedule. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetEffectiveSteps(Double,Int32)` | Gets the number of denoising steps to perform for a given strength. |
| `GetStartNoiseLevel(Double)` | Gets the noise level (alpha_bar) at the starting timestep. |
| `GetStartTimestep(Double)` | Gets the starting timestep for a given strength. |
| `TruncateSchedule([],Double)` | Truncates a full timestep schedule to start from the strength-determined point. |

