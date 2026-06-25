---
title: "VERSAAlgorithm<T, TInput, TOutput>"
description: "Implementation of VERSA (Versatile and Efficient Few-shot Learning) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of VERSA (Versatile and Efficient Few-shot Learning) for few-shot learning.

## For Beginners

VERSA works like a "classifier factory":

**How it works:**

1. Feed support examples through the feature extractor to get features
2. Aggregate features per class (e.g., compute class means)
3. Feed aggregated features through the amortization network
4. The amortization network outputs classifier weights (instantly!)
5. Use those weights to classify query examples

**Analogy:**
Imagine a car factory that can produce custom cars:

- You describe what you want (support examples)
- The factory immediately produces a car matching your specs (classifier)
- No iterative refinement needed - it's a single manufacturing step

**Compared to other methods:**

- MAML: "Let me practice on these examples for 5 rounds" (iterative)
- R2-D2: "Let me solve a math equation" (closed-form but still per-task)
- VERSA: "I've been trained to instantly know what classifier you need" (amortized)

**Key benefit:** Amortization generalizes across tasks, so VERSA doesn't need to
solve each task from scratch - it recognizes patterns in support sets.

## How It Works

VERSA uses an amortization network that takes aggregated support set features and produces
task-specific classifier parameters in a single forward pass. No inner-loop optimization
is required, making adaptation extremely fast.

**Algorithm - VERSA:**

Reference: Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., & Turner, R. E. (2019).
Meta-Learning Probabilistic Inference for Prediction. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VERSAAlgorithm(VERSAOptions<,,>)` | Initializes a new VERSA meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using amortized inference (single forward pass). |
| `AmortizeClassifier()` | Produces classifier weights from support features using the amortization network. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the amortization network. |
| `InitializeAmortizationNetwork` | Initializes the amortization network parameters with small random values. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for VERSA. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_amortizationParams` | Amortization network parameters that produce classifier weights from support features. |

