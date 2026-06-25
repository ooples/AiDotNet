---
title: "SimplePreferenceOptimization<T, TInput, TOutput>"
description: "Implements Simple Preference Optimization (SimPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Simple Preference Optimization (SimPO) for fine-tuning.

## For Beginners

SimPO is like DPO but simpler and more memory efficient.
It doesn't need to keep a frozen copy of the original model, which saves GPU memory.
Instead, it uses the average log probability of responses and a target reward margin.

## How It Works

SimPO is a reference-free preference optimization method that outperforms DPO
while being more computationally efficient (no need for a reference model).

Key differences from DPO:

1. Uses average log probability instead of sum (length-normalized)
2. No reference model needed
3. Adds a target reward margin (gamma) for stability

The SimPO loss function is:
L_SimPO = -log σ(β/|y| * (log π(y_w|x) - log π(y_l|x)) - γ)
where γ is the target margin and |y| denotes response length for normalization.

Original paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
by Meng et al. (2024) - NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimplePreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of SimPO fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MethodName` |  |
| `RequiresReferenceModel` |  |
| `RequiresRewardModel` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAverageLogProbability(IFullModel<,,>,,)` | Computes the average log probability of an output (length-normalized). |
| `ComputeSimPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,Double,CancellationToken)` | Computes the SimPO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `GetOutputLength()` | Gets the length of an output for normalization. |

