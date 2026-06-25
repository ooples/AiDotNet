---
title: "MetaLoRAAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-LoRA: Low-Rank Adaptation for Meta-Learning (2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-LoRA: Low-Rank Adaptation for Meta-Learning (2024).

## How It Works

Meta-LoRA applies Low-Rank Adaptation (LoRA) to the meta-learning inner loop.
Instead of adapting all d model parameters per task (as in MAML), it meta-learns
r low-rank basis vectors {v_1, ..., v_r} in parameter space and only adapts
r scalar coefficients {c_1, ..., c_r} during the inner loop.

**Algorithm:**

**Advantages over MAML:**
Inner loop only adapts r parameters instead of d, making adaptation much cheaper
for large models. The low-rank constraint also acts as an implicit regularizer,
preventing overfitting on few-shot support sets.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAdaptedParams(Vector<>,Double[])` | Computes adapted parameters: θ' = θ_base + (α/r) * Σ c_i * v_i. |
| `ComputeLoRALoss(TaskBatch<,,>)` | Loss function for SPSA-based updates of basis vectors and initial coefficients. |
| `DotProductWithBasis(Vector<>,Int32)` | Computes dot product of a gradient vector with the i-th basis vector. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_loraBasis` | Low-rank basis vectors stored as a flat vector of length rank * paramDim. |
| `_loraCoeffInit` | Meta-learned initial coefficients for the low-rank basis (length = rank). |

