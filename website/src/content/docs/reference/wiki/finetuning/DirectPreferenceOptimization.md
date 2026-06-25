---
title: "DirectPreferenceOptimization<T, TInput, TOutput>"
description: "Implements Direct Preference Optimization (DPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Direct Preference Optimization (DPO) for fine-tuning.

## For Beginners

DPO learns from pairs of responses where one is preferred over
the other. Instead of training a reward model first (like RLHF), DPO directly adjusts
the model to make preferred responses more likely and rejected responses less likely.

## How It Works

DPO is a popular preference optimization method that directly optimizes the model
using preference pairs without requiring a separate reward model.

The DPO loss function is:
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
where y_w is the chosen response, y_l is the rejected response, and β controls
the strength of preference learning.

Original paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
by Rafailov et al. (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DirectPreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of DPO fine-tuning. |

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
| `ComputeDPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the DPO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

