---
title: "FreeNoiseModule<T>"
description: "FreeNoise module for tuning-free longer video generation via noise rescheduling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Acceleration`

FreeNoise module for tuning-free longer video generation via noise rescheduling.

## For Beginners

The FreeNoise module rearranges noise patterns to enable longer video generation from models trained on short clips. It is a simple, training-free technique that extends video length without quality loss.

## How It Works

**References:**

- Paper: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling" (Qiu et al., 2024)

FreeNoise enables generating longer videos from short-video diffusion models without any
fine-tuning. The key idea is noise rescheduling: instead of using independent random noise
for all frames, FreeNoise constructs temporally correlated noise by:

1. Generating a base noise sequence for the window size
2. Shifting and blending noise for extended frames
3. Using window-based attention with shared noise patterns

This maintains temporal consistency across windows while extending generation length.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FreeNoiseModule(Int32,Int32,Double,Nullable<Int32>)` | Initializes a new FreeNoise module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlendRatio` | Gets the blend ratio between shifted and fresh noise. |
| `NoiseShiftStride` | Gets the noise shift stride. |
| `WindowSize` | Gets the window size for noise generation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateRescheduledNoise(Int32,Int32[])` | Generates temporally correlated noise for a target number of frames. |
| `GetBaseNoise` | Gets a copy of the stored base noise from the last generation. |
| `Reset` | Resets the module state. |
| `SampleGaussian` | Samples a value from the standard normal distribution using Box-Muller transform. |

