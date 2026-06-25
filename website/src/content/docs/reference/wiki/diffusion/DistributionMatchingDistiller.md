---
title: "DistributionMatchingDistiller<T>"
description: "Distribution Matching Distillation (DMD) trainer for single-step generation via distribution alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Distribution Matching Distillation (DMD) trainer for single-step generation via distribution alignment.

## For Beginners

DMD trains a fast single-step generator by ensuring its outputs "look like"
the outputs of the slow multi-step teacher. It checks both individual quality (regression loss:
"does each image look right?") and overall diversity (distribution loss: "does the collection of
generated images look like what the teacher would produce?").

## How It Works

DMD distills a diffusion model into a single-step generator by matching the output distribution
of the generator to that of the teacher diffusion model. Combines a regression loss (MSE between
generator output and teacher's denoised output) with a distribution matching loss that ensures
the generator's output distribution matches the teacher's. DMD2 improves this by removing the
need for a GAN discriminator.

Reference: Yin et al., "One-step Diffusion with Distribution Matching Distillation", CVPR 2024;
Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistributionMatchingDistiller(IDiffusionModel<>,Double,Double,Double)` | Initializes a new DMD trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistributionWeight` | Gets the distribution matching loss weight. |
| `RegressionWeight` | Gets the regression loss weight. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistributionLoss(Vector<>,Vector<>)` | Computes the distribution matching loss using score differences. |
| `ComputeRegressionLoss(Vector<>,Vector<>)` | Computes the regression loss between generator output and teacher's denoised output. |
| `ComputeTotalLoss(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the combined DMD loss (regression + distribution matching). |

