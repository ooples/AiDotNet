---
title: "StatisticalRejectionSampling<T, TInput, TOutput>"
description: "Implements Statistical Rejection Sampling Optimization (RSO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Statistical Rejection Sampling Optimization (RSO) for fine-tuning.

## For Beginners

RSO takes a list of responses ranked from best to worst
and intelligently picks pairs to learn from. Instead of just using best vs worst,
it uses statistical sampling to create more informative training examples.

## How It Works

RSO uses rejection sampling to create high-quality training pairs from ranked outputs.
It samples pairs based on the statistical properties of the ranking to create
better preference data.

Original paper: "Statistical Rejection Sampling Improves Preference Optimization"
by Liu et al. (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StatisticalRejectionSampling(FineTuningOptions<>)` | Initializes a new instance of RSO fine-tuning. |

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
| `ComputePairLoss(IFullModel<,,>,,,,Double)` | Computes the DPO-style loss for a preference pair. |
| `ComputeRSOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the RSO loss for a batch. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `SamplePairsWithRejection([])` | Samples pairs from ranked outputs using statistical rejection sampling. |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

