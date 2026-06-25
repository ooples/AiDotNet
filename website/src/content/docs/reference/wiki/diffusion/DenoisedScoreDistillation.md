---
title: "DenoisedScoreDistillation<T>"
description: "Denoised Score Distillation (DSD) for artifact-free 3D generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Denoised Score Distillation (DSD) for artifact-free 3D generation.

## For Beginners

Regular SDS compares noise predictions, which can be imprecise.
DSD takes a different approach — it fully generates a clean reference image and compares
it to what the 3D model renders. This is like having the teacher paint a complete example
for every critique, giving much clearer feedback to the 3D model.

## How It Works

DSD addresses the mode-seeking bias of SDS by using a fully denoised reference image
instead of comparing noisy predictions. At each optimization step, it runs the full
diffusion denoising chain to get a clean reference, then uses the difference between
this reference and the rendered view as the gradient. While more expensive per step,
DSD produces significantly sharper and more diverse 3D results.

Reference: Hertz et al., "Denoised Score Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DenoisedScoreDistillation(IDiffusionModel<>,Double,Int32)` | Initializes a new DSD instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DenoisingSteps` | Gets the number of full denoising steps for reference generation. |
| `GuidanceScale` | Gets the guidance scale. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Double)` | Computes the DSD gradient from a fully denoised reference. |

