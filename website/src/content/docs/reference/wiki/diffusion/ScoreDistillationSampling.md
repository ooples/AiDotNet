---
title: "ScoreDistillationSampling<T>"
description: "Score Distillation Sampling (SDS) for text-to-3D and generator optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Score Distillation Sampling (SDS) for text-to-3D and generator optimization.

## For Beginners

SDS uses a 2D image generator as a "critic" for 3D models. It renders
the 3D model from a random angle, asks the 2D model "how could this image be improved?",
and uses that feedback to update the 3D model. Repeating this from many angles creates
a 3D object that looks good from every direction.

## How It Works

SDS computes gradients from a pretrained 2D diffusion model to optimize a 3D representation
(NeRF, mesh, etc.). It adds noise to rendered views, asks the diffusion model to predict
the noise, and uses the difference between added and predicted noise as a gradient signal.
This enables text-to-3D generation without any 3D training data.

Reference: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScoreDistillationSampling(IDiffusionModel<>,Double,Double,Double,Double)` | Initializes a new SDS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the classifier-free guidance scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Double)` | Computes the SDS gradient for a rendered view. |
| `SampleTimestep(Int32,Random)` | Samples a random timestep within the configured range. |

