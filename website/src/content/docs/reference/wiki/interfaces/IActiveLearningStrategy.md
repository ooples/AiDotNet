---
title: "IActiveLearningStrategy<T>"
description: "Defines a strategy for active learning that selects the most informative samples for labeling from a pool of unlabeled data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a strategy for active learning that selects the most informative samples
for labeling from a pool of unlabeled data.

## For Beginners

Active learning helps when labeling data is expensive or time-consuming.
Instead of randomly selecting samples to label, active learning intelligently picks the samples
that would be most helpful for training the model. This can dramatically reduce the number of
labels needed while achieving similar or better performance.

## How It Works

**Common strategies include:**

**Typical Usage Flow:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this active learning strategy. |
| `UseBatchDiversity` | Gets or sets whether to use batch-mode selection that considers diversity among selected samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` | Computes informativeness scores for all samples in the unlabeled pool. |
| `GetSelectionStatistics` | Gets statistics about the most recent sample selection. |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` | Selects the most informative samples from the unlabeled pool for labeling. |

