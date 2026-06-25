---
title: "RobustDirectPreferenceOptimization<T, TInput, TOutput>"
description: "Implements Robust Direct Preference Optimization (RDPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Robust Direct Preference Optimization (RDPO) for fine-tuning.

## For Beginners

Real-world preference data often contains mistakes -
annotators sometimes disagree or make errors. RDPO handles this by being more
cautious about uncertain preferences and focusing on high-confidence examples.

## How It Works

RDPO extends DPO to handle noisy or inconsistent preference labels by incorporating
label confidence and noise-aware training.

Key features:

- Confidence-weighted preference learning
- Robust to label noise and annotator disagreement
- Uses sample weights to down-weight uncertain examples

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RobustDirectPreferenceOptimization(FineTuningOptions<>)` | Initializes a new instance of RDPO fine-tuning. |

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
| `ComputeRDPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the RDPO loss for a batch with robustness to label noise. |
| `ComputeRobustnessWeight(Double,Double)` | Computes the robustness weight for a sample based on model confidence and label confidence. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

