---
title: "SupervisedFineTuning<T, TInput, TOutput>"
description: "Implements Supervised Fine-Tuning (SFT) - the foundational fine-tuning method."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Supervised Fine-Tuning (SFT) - the foundational fine-tuning method.

## For Beginners

SFT is like teaching by example. You show the model many
examples of correct input-output pairs, and it learns to produce similar outputs.
For instance, you might train a model on high-quality question-answer pairs to make
it better at answering questions.

## How It Works

SFT trains a model on labeled input-output pairs using standard supervised learning.
This is the most straightforward fine-tuning approach and serves as the foundation
for more advanced methods like RLHF or DPO.

**Use Cases:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SupervisedFineTuning(FineTuningOptions<>)` | Initializes a new instance of SFT with default loss function. |
| `SupervisedFineTuning(FineTuningOptions<>,ILossFunction<>)` | Initializes a new instance of SFT with a custom loss function. |

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
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `IsCorrectPrediction(Vector<>,Vector<>)` | Checks if a prediction matches the target (for accuracy calculation). |
| `ProcessBatchAsync(IFullModel<,,>,FineTuningData<,,>,ILossFunction<>,)` | Processes a single batch of training data. |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

