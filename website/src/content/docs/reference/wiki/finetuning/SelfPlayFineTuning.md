---
title: "SelfPlayFineTuning<T, TInput, TOutput>"
description: "Implements Self-Play Fine-Tuning (SPIN) for model improvement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Self-Play Fine-Tuning (SPIN) for model improvement.

## For Beginners

SPIN is like a model playing chess against itself
to get better. The current model learns to prefer real human responses over
its own generated responses, iteratively improving without new labeled data.

## How It Works

SPIN uses self-play to improve models without additional human-labeled data.
The model plays against previous versions of itself, learning to distinguish
its own outputs from ground truth.

Key features:

- No additional human labels needed after initial SFT
- Iterative improvement through self-play
- Uses DPO-style objective with model's own outputs as negatives

Original paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"
by Chen et al. (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfPlayFineTuning(FineTuningOptions<>)` | Initializes a new instance of SPIN fine-tuning. |

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
| `ComputeSPINLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,CancellationToken)` | Computes the SPIN loss for a batch. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

