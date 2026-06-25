---
title: "DMD2Model<T>"
description: "Distribution Matching Distillation v2 (DMD2) for single-step high-fidelity generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Distribution Matching Distillation v2 (DMD2) for single-step high-fidelity generation.

## For Beginners

DMD2 trains a student model to produce images that match the
distribution of a teacher (full diffusion model) in a single step. It's like having
a talented student who can paint a complete picture in one stroke, trained by watching
a master who paints in many careful strokes.

## How It Works

DMD2 improves upon DMD by introducing a regression loss alongside the distribution
matching loss, eliminating the need for an expensive GAN discriminator. Achieves
state-of-the-art single-step FID on ImageNet while being simpler to train than ADD.

Reference: Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DMD2Model(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new DMD2 model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

