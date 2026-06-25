---
title: "SDCLAlgorithm<T, TInput, TOutput>"
description: "Implementation of SDCL: Self-Distillation Collaborative Learning for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of SDCL: Self-Distillation Collaborative Learning for meta-learning.

## How It Works

SDCL applies knowledge distillation within the meta-learning loop. A teacher model
(EMA of student parameters) provides soft targets that regularize the adapted student.
The distillation loss (symmetric KL divergence between teacher and student output
distributions) stabilizes adaptation and prevents overfitting to the few support examples.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeDistillationGradient(Vector<>,Vector<>,)` | Computes the distillation gradient: ∂KL(teacher \|\| student) / ∂θ_student. |
| `ComputeKLDivergence(Vector<>,Vector<>)` | Computes symmetric KL divergence between two output vectors treated as soft distributions. |
| `GetPredictionWithParams(Vector<>,)` | Gets a prediction by temporarily setting model parameters and running forward pass. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_teacherParams` | Teacher model parameters (EMA of student). |

