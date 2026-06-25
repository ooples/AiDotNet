---
title: "VariationalScoreDistillation<T>"
description: "Variational Score Distillation (VSD) for high-fidelity text-to-3D generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Variational Score Distillation (VSD) for high-fidelity text-to-3D generation.

## For Beginners

SDS sometimes produces washed-out or over-saturated 3D models. VSD
fixes this by training a second model that learns what the 3D model's renders look like.
By comparing the "ideal" score (from the pretrained model) with the "current" score (from
the adapted model), it gives more precise, less noisy feedback for 3D optimization.

## How It Works

VSD improves upon SDS by reducing the over-saturation and over-smoothing artifacts. It
maintains a separate LoRA-adapted diffusion model trained on rendered views and uses the
difference between the pretrained model's score and the LoRA model's score as the gradient.
This effectively performs variational inference in the diffusion model's data space.

Reference: Wang et al., "ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation
with Variational Score Distillation", NeurIPS 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalScoreDistillation(IDiffusionModel<>,Double,Double,Int32)` | Initializes a new VSD instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceScale` | Gets the guidance scale for the pretrained model. |
| `LoRARank` | Gets the LoRA rank for the particle model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>,Double)` | Computes the VSD gradient using both pretrained and particle model scores. |
| `ComputeParticleLoss(Vector<>,Vector<>)` | Computes the LoRA training loss for the particle model. |

