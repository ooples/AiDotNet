---
title: "KahnemanTverskyOptimization<T, TInput, TOutput>"
description: "Implements Kahneman-Tversky Optimization (KTO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Kahneman-Tversky Optimization (KTO) for fine-tuning.

## For Beginners

KTO is based on how humans actually make decisions.
People feel losses more strongly than equivalent gains (loss aversion).
KTO uses this insight to train models - making them more careful about avoiding
bad outputs than they are eager to produce good ones.

## How It Works

KTO applies prospect theory from behavioral economics to preference learning.
Unlike DPO, it doesn't require paired data - each example is independently labeled
as desirable or undesirable.

Key features:

- Doesn't require paired preference data
- Uses loss aversion (undesirable weight typically higher than desirable weight)
- More sample efficient for imbalanced datasets

Original paper: "KTO: Model Alignment as Prospect Theoretic Optimization"
by Ethayarajh et al. (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KahnemanTverskyOptimization(FineTuningOptions<>)` | Initializes a new instance of KTO fine-tuning. |

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
| `ComputeBatchKLEstimate(IFullModel<,,>,FineTuningData<,,>)` | Computes the KL divergence estimate for the batch. |
| `ComputeKTOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,Double,Double,Double,CancellationToken)` | Computes the KTO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

