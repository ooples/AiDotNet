---
title: "ContrastivePreferenceOptimization<T, TInput, TOutput>"
description: "Implements Contrastive Preference Optimization (CPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Contrastive Preference Optimization (CPO) for fine-tuning.

## For Beginners

CPO trains the model by directly comparing preferred
and rejected responses. It's simpler than DPO because it doesn't need to keep
a frozen copy of the original model.

## How It Works

CPO is a variant of DPO that focuses on contrastive learning between chosen
and rejected responses without requiring a reference model, similar to ORPO
but with a different loss formulation.

The CPO loss directly maximizes the gap between chosen and rejected outputs:
L_CPO = -log σ(β * (log π(y_w|x) - log π(y_l|x)))

Key advantages:

- No reference model needed (saves memory)
- Simpler implementation
- Good for scenarios where you want the model to deviate from its initial behavior

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastivePreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of CPO fine-tuning. |

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
| `ComputeCPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the CPO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

