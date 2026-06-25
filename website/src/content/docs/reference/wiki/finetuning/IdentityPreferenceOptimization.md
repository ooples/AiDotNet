---
title: "IdentityPreferenceOptimization<T, TInput, TOutput>"
description: "Implements Identity Preference Optimization (IPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Identity Preference Optimization (IPO) for fine-tuning.

## For Beginners

IPO is like DPO but more stable. It learns preferences
without being too aggressive, which helps prevent the model from becoming overconfident
or forgetting how to generate diverse responses.

## How It Works

IPO improves upon DPO by addressing overfitting issues through a different loss function
that doesn't push the rejected response probability to zero.

The IPO loss function is:
L_IPO = (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2β))²
This squared loss form provides better gradients and prevents collapse.

Original paper: "A General Theoretical Paradigm to Understand Learning from Human Preferences"
by Azar et al. (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IdentityPreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of IPO fine-tuning. |

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
| `ComputeIPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the IPO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

