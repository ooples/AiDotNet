---
title: "ConsistencyTrainingTrainer<T>"
description: "Trainer for consistency training from scratch without a pretrained teacher model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for consistency training from scratch without a pretrained teacher model.

## For Beginners

While consistency distillation needs a "teacher" model to learn from,
consistency training learns on its own. It discovers that different amounts of noise added
to the same image should all map back to the same clean image. This is like learning to
recognize a face from photos taken at different exposure levels — they're all the same face.

## How It Works

Unlike consistency distillation (which requires a pretrained teacher), consistency training
learns the consistency function directly from data using a self-consistency loss. The model
learns that mapping any point along the same ODE trajectory should produce the same output,
without needing teacher guidance. Uses a progressive schedule that increases discretization
steps during training.

Reference: Song et al., "Improved Techniques for Training Consistency Models", ICML 2024 (iCT);
Song and Dhariwal, "Consistency Models", ICML 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConsistencyTrainingTrainer(Double,Int32,Int32,Double,Double)` | Initializes a new consistency training trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentStep` | Gets the current training step. |
| `InitialDiscretizationSteps` | Gets the initial discretization step count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeConsistencyLoss(Vector<>,Vector<>)` | Computes the self-consistency loss for a training batch. |
| `GetAdaptiveEMADecay(Int32)` | Computes the adaptive EMA decay rate based on current discretization. |
| `GetCurrentDiscretizationSteps(Int32)` | Gets the current number of discretization steps based on training progress. |
| `GetSigma(Int32,Int32)` | Gets the sigma (noise level) for a given discretization index. |
| `Step` | Advances the training step counter. |

