---
title: "RankResponsesHumanFeedback<T, TInput, TOutput>"
description: "Implements Rank Responses to Align Human Feedback (RRHF) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Rank Responses to Align Human Feedback (RRHF) for fine-tuning.

## For Beginners

RRHF learns from a full ranking of responses,
not just pairs. If you have 5 responses ranked from best to worst, RRHF
uses all that ranking information to train the model more efficiently.

## How It Works

RRHF aligns language models by ranking multiple responses and learning from
the ranking order rather than just pairwise preferences.

Key features:

- Uses full ranking information, not just pairs
- More efficient use of human feedback
- Combines SFT and ranking objectives

Original paper: "RRHF: Rank Responses to Align Language Models with Human Feedback
without tears" by Yuan et al. (2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RankResponsesHumanFeedback(FineTuningOptions<>)` | Initializes a new instance of RRHF fine-tuning. |

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
| `ComputeKendallTau(Double[])` | Computes Kendall's tau correlation coefficient for a ranking. |
| `ComputeRRHFLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,Double,CancellationToken)` | Computes the RRHF loss for a batch. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

