---
title: "PairwiseRankingOptimization<T, TInput, TOutput>"
description: "Implements Pairwise Ranking Optimization (PRO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Pairwise Ranking Optimization (PRO) for fine-tuning.

## For Beginners

PRO treats model alignment as a ranking problem.
Given multiple responses, it learns to assign higher scores to better responses,
using techniques from information retrieval and recommendation systems.

## How It Works

PRO optimizes models using pairwise comparisons from ranked lists,
similar to learning-to-rank algorithms used in search engines.

Key features:

- Uses pairwise ranking loss (similar to RankNet)
- Can leverage ranking information from multiple annotators
- Robust to noise through margin-based learning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PairwiseRankingOptimization(FineTuningOptions<>)` | Initializes a new instance of PRO fine-tuning. |

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
| `ComputePROLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,Double,CancellationToken)` | Computes the PRO loss for a batch. |
| `ComputePairwiseRankingLoss(IFullModel<,,>,,,,Double,Double)` | Computes the pairwise ranking loss (RankNet-style). |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

