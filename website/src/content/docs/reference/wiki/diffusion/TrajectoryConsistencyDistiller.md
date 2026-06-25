---
title: "TrajectoryConsistencyDistiller<T>"
description: "Trainer for Trajectory Consistency Distillation (TCD) with trajectory-aware loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for Trajectory Consistency Distillation (TCD) with trajectory-aware loss.

## For Beginners

Standard consistency distillation checks if the student gets
the right answer at each individual step. TCD also checks if the student's answers
are consistent with each other across the whole journey — like checking that a
student's essay tells a coherent story, not just that each sentence is correct.

## How It Works

TCD extends standard consistency distillation by considering the entire ODE trajectory
rather than individual timestep pairs. The trajectory-aware loss ensures the student
is self-consistent across the entire denoising path, producing higher quality at
any step count.

Reference: Zheng et al., "Trajectory Consistency Distillation", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrajectoryConsistencyDistiller(IDiffusionModel<>,Double,Double)` | Initializes a new TCD trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EMADecay` | Gets the EMA decay rate. |
| `StochasticEta` | Gets the stochastic noise injection strength. |
| `Teacher` | Gets the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeTrajectoryLoss(Vector<>[],Vector<>[])` | Computes the trajectory consistency loss across multiple timestep pairs. |

