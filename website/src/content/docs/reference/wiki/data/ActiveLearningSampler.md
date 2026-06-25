---
title: "ActiveLearningSampler<T>"
description: "A sampler for active learning that selects the most informative samples for labeling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler for active learning that selects the most informative samples for labeling.

## For Beginners

In active learning, you don't have labels for all data.
This sampler helps you decide which unlabeled samples to ask an expert to label:

- **Uncertainty sampling**: Ask about samples the model is unsure about
- **Diversity sampling**: Ask about samples that are different from what you've seen
- **Expected model change**: Ask about samples that would change the model most

This can dramatically reduce labeling costs (50-90% less labels needed)!

## How It Works

ActiveLearningSampler implements uncertainty sampling and other active learning
strategies to select unlabeled samples that would be most valuable to label.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ActiveLearningSampler(Int32,ActiveLearningStrategy,Double,Nullable<Int32>)` | Initializes a new instance of the ActiveLearningSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LabeledCount` | Gets the number of labeled samples. |
| `Length` |  |
| `UnlabeledCount` | Gets the number of unlabeled samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |
| `MarkAsLabeled(IEnumerable<Int32>)` | Marks multiple samples as labeled. |
| `MarkAsLabeled(Int32)` | Marks a sample as labeled. |
| `SelectForLabeling(Int32)` | Selects the most informative unlabeled samples for labeling. |
| `UpdateUncertainties(IReadOnlyList<Int32>,IReadOnlyList<>)` | Batch updates uncertainty scores. |
| `UpdateUncertainty(Int32,)` | Updates the uncertainty score for a sample. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

