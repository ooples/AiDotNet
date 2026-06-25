---
title: "UnifiedDistillationSampling<T>"
description: "Unified Distillation Sampling (UDS) framework unifying SDS, VSD, CSD, and ISM variants."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Unified Distillation Sampling (UDS) framework unifying SDS, VSD, CSD, and ISM variants.

## For Beginners

Different score distillation methods (SDS, VSD, etc.) each have
strengths and weaknesses. UDS is a unified framework that can act as any of them by
adjusting its settings. It's like a Swiss Army knife for 3D generation — you can dial
in the exact balance of quality, speed, and diversity that you need.

## How It Works

UDS provides a unified theoretical framework that encompasses SDS, VSD, CSD, ISM, and DSD
as special cases. It parameterizes the score distillation loss with configurable weights
for different gradient components, allowing smooth interpolation between methods and
enabling new hybrid approaches.

Reference: Katzir et al., "A Unified Framework for Score Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnifiedDistillationSampling(IDiffusionModel<>,Double,Double,Double,Double)` | Initializes a new UDS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NoiseWeight` | Gets the weight for the noise baseline component. |
| `ParticleWeight` | Gets the weight for the particle/LoRA model score component. |
| `PretrainedWeight` | Gets the weight for the pretrained model score component. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Vector<>,Double)` | Computes the unified gradient combining all score components. |

