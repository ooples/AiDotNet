---
title: "MAMLPlusPlusAlgorithm<T, TInput, TOutput>"
description: "Implementation of MAML++ (How to Train Your MAML) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of MAML++ (How to Train Your MAML) for few-shot learning.

## For Beginners

MAML++ is the "industrial strength" version of MAML:

**Problems with vanilla MAML:**

1. Training is unstable (loss can explode randomly)
2. One learning rate doesn't work well for all adaptation steps
3. Second-order gradients are noisy early in training
4. Batch normalization statistics become stale during inner loop

**MAML++ solutions:**

1. Multi-step loss: Check performance at EVERY step, not just the last
2. Per-step learning rates: Each step gets its own rate, learned during training
3. Derivative-order annealing: Start simple, gradually get more precise
4. Per-step batch norm: Keep separate statistics for each step

**Analogy:** If MAML is a car, MAML++ adds:

- Anti-lock brakes (stability fixes)
- Cruise control (per-step learning rates)
- A GPS that updates gradually (derivative-order annealing)
- Multiple speedometers (per-step batch norm)

## How It Works

MAML++ is a production-hardened MAML that addresses training instabilities through:

- Multi-Step Loss (MSL): Supervise every inner-loop step, not just the final one
- Per-Step Learning Rates (LSLR): Each adaptation step has its own learnable learning rate
- Derivative-Order Annealing: Gradually transition from first-order to second-order gradients
- Per-Step Batch Normalization: Separate BN statistics for each adaptation step
- Cosine Annealing: Learning rate schedule for the outer loop

**Algorithm - MAML++:**

Reference: Antoniou, A., Edwards, H., & Storkey, A. (2019).
How to Train Your MAML. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MAMLPlusPlusAlgorithm(MAMLPlusPlusOptions<,,>)` | Initializes a new instance of the MAML++ algorithm. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |
| `PerStepLearningRates` | Gets the current per-step learning rates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using MAML++ per-step learning rates. |
| `ComputeCosineAnnealedLR` | Computes the cosine-annealed outer learning rate for the current iteration. |
| `EvaluateTaskLoss(IMetaLearningTask<,,>,Vector<>)` | Evaluates the query loss for a single task after running the full inner loop. |
| `GetMultiStepWeights` | Gets the multi-step loss weights for each adaptation step plus the final query step. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step with MAML++ enhancements. |
| `ShouldUseFirstOrder` | Determines whether to use first-order gradients based on derivative-order annealing. |
| `UpdatePerStepLearningRates(TaskBatch<,,>,Vector<>,Double)` | Updates per-step learning rates using finite differences. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_perStepLearningRates` | Per-step learning rates that are meta-learned during training. |

