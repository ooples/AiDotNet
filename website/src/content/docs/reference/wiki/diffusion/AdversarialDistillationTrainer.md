---
title: "AdversarialDistillationTrainer<T>"
description: "Trainer for Adversarial Diffusion Distillation (ADD) as used in SD/SDXL Turbo."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for Adversarial Diffusion Distillation (ADD) as used in SD/SDXL Turbo.

## For Beginners

ADD uses two teachers: (1) a diffusion model that ensures the
student captures the right image structure, and (2) a discriminator that ensures the
output looks realistic. This dual supervision produces remarkably good single-step
images — the student learns both "what" to generate and "how real" it should look.

## How It Works

ADD combines a diffusion denoising loss with an adversarial loss from a pretrained
discriminator. The student generates a sample in 1-4 steps, a discriminator evaluates
realism, and the diffusion teacher provides structure guidance. This dual-loss approach
produces the highest quality single-step generators.

Reference: Sauer et al., "Adversarial Diffusion Distillation", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialDistillationTrainer(IDiffusionModel<>,Double,Double)` | Initializes a new adversarial distillation trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialWeight` | Gets the adversarial loss weight. |
| `DiffusionWeight` | Gets the diffusion loss weight. |
| `Teacher` | Gets the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeADDLoss(Vector<>,Vector<>,)` | Computes the combined ADD loss. |

