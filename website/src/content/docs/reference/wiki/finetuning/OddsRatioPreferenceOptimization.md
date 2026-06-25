---
title: "OddsRatioPreferenceOptimization<T, TInput, TOutput>"
description: "Implements Odds Ratio Preference Optimization (ORPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Odds Ratio Preference Optimization (ORPO) for fine-tuning.

## For Beginners

ORPO is a clever method that learns both to produce
good outputs (like SFT) and to prefer good outputs over bad ones (like DPO)
at the same time. It's simpler because it doesn't need a frozen reference model.

## How It Works

ORPO combines SFT and preference optimization in a single training objective.
It doesn't require a reference model, making it simpler and more memory efficient.

The ORPO loss combines:

1. SFT loss on chosen outputs: -log P(y_w|x)
2. Odds ratio loss: -λ * log(odds(y_w) / odds(y_l))

where odds(y) = P(y|x) / (1 - P(y|x))

Original paper: "ORPO: Monolithic Preference Optimization without Reference Model"
by Hong et al. (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OddsRatioPreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of ORPO fine-tuning. |

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
| `ComputeLogOdds(Double)` | Computes the log odds from log probability: log(P/(1-P)). |
| `ComputeORPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the ORPO loss for a batch and updates model parameters. |
| `ComputeOdds(Double)` | Computes the odds from log probability: P/(1-P). |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

