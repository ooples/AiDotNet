---
title: "RefinerStage<T>"
description: "Refiner stage for late-stage noise-add-then-denoise detail improvement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Alignment`

Refiner stage for late-stage noise-add-then-denoise detail improvement.

## For Beginners

Imagine you've painted a picture but want to add more detail. The
refiner slightly blurs the image (adds noise), then carefully sharpens it back. This
re-processing enhances fine details like skin texture, fabric patterns, and small objects
that the main generation pass might have missed.

## How It Works

The refiner stage takes a nearly-finished generated image, adds a small amount of noise
back, and re-denoises using a specialized refiner model (or the same model with different
conditioning). This late-stage refinement enhances fine details, textures, and overall
coherence. Used in SDXL's two-stage pipeline and other multi-stage generation approaches.

Reference: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution
Image Synthesis", ICLR 2024 (refiner stage)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RefinerStage(IDiffusionModel<>,Double,Double,Int32)` | Initializes a new refiner stage. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the classifier-free guidance scale for the refiner. |
| `RefinerSteps` | Gets the number of denoising steps the refiner uses. |
| `Strength` | Gets the noise strength for refinement (0.0 = no change, 1.0 = full re-generation). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoiseForRefinement(Vector<>,Vector<>)` | Adds noise to a latent at the specified strength level. |
| `ComputeRefinementDelta(Vector<>,Vector<>)` | Computes the refinement quality improvement estimate. |
| `GetStartTimestep(Int32)` | Computes the starting timestep for the refiner based on strength. |

