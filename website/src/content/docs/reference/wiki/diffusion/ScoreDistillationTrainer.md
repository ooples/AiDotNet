---
title: "ScoreDistillationTrainer<T>"
description: "Trainer for Score Distillation Sampling (SDS) and its variants for generator training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for Score Distillation Sampling (SDS) and its variants for generator training.

## For Beginners

Score distillation asks a pretrained diffusion model "how could
this generated image be improved?" and uses the answer to train the generator. It's
like having an art critic (the diffusion model) provide feedback that the student
generator uses to improve. VSD and CSD are refined versions that give better feedback.

## How It Works

SDS uses the score function (gradient of the log-probability) from a pretrained
diffusion model to guide a generator network. The generator produces samples, noise
is added, and the diffusion model's score provides the training gradient. Supports
Variational Score Distillation (VSD) and Classifier Score Distillation (CSD) variants.

Reference: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023 (SDS);
Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation", NeurIPS 2023 (VSD)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScoreDistillationTrainer(IDiffusionModel<>,Double,Double,Double)` | Initializes a new score distillation trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the guidance scale for score computation. |
| `MaxTimestep` | Gets the maximum timestep. |
| `MinTimestep` | Gets the minimum timestep. |
| `Teacher` | Gets the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSDSGradient(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the SDS gradient for a generator's output. |

