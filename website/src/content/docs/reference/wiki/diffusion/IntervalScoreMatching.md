---
title: "IntervalScoreMatching<T>"
description: "Interval Score Matching (ISM) for improved 3D score distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Interval Score Matching (ISM) for improved 3D score distillation.

## For Beginners

SDS gradients are noisy because they compare random noise with the
model's prediction. ISM is cleverer — it uses a deterministic process to add and remove
noise over a small interval. This gives much smoother, more reliable feedback for 3D
optimization, resulting in cleaner 3D models with fewer artifacts.

## How It Works

ISM reduces the variance of SDS gradients by using deterministic DDIM inversion between
two timesteps instead of random noise injection. Given a rendered view, ISM deterministically
noises it to timestep t2, then denoises to timestep t1, and uses the difference as the
gradient signal. This "interval" approach produces much lower-variance gradients than SDS.

Reference: Liang et al., "LucidDreamer: Towards High-Fidelity Text-to-3D Generation via
Interval Score Matching", CVPR 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IntervalScoreMatching(IDiffusionModel<>,Double,Int32)` | Initializes a new ISM instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the guidance scale. |
| `IntervalSteps` | Gets the number of DDIM steps in the interval. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Vector<>,Double)` | Computes the ISM gradient between two timestep points. |

