---
title: "ConsistencyDistillationTrainer<T>"
description: "Trainer for consistency distillation from a pretrained diffusion model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for consistency distillation from a pretrained diffusion model.

## For Beginners

Given a pretrained diffusion model (teacher), this trainer
creates a fast student model. The teacher shows the student what the clean image
should look like from any noise level, and the student learns to jump directly
to that clean image in one step.

## How It Works

Implements the consistency distillation procedure where a student model learns to map
any point on the ODE trajectory directly to the trajectory's origin (clean data).
The teacher provides target outputs via one-step ODE evaluation, and the student
learns to match these targets with a consistency loss.

Reference: Song et al., "Consistency Models", ICML 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyDistillationTrainer(IDiffusionModel<>,Double,Int32)` | Initializes a new consistency distillation trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EMADecay` | Gets the EMA decay rate. |
| `NumTimestepBins` | Gets the number of timestep discretization bins. |
| `Teacher` | Gets the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeConsistencyLoss(Vector<>,Vector<>)` | Computes the consistency distillation loss for a training batch. |

