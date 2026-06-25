---
title: "RandomSampling<T>"
description: "Implements random sampling for active learning (baseline strategy)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning`

Implements random sampling for active learning (baseline strategy).

## For Beginners

Random sampling is the simplest active learning strategy.
It randomly selects samples from the unlabeled pool without considering model predictions
or any informativeness measure. This serves as a baseline for comparing other strategies.

## How It Works

**When to use:**

**Complexity:** O(n) for selection where n is the pool size.

**Reference:** Standard baseline in active learning literature.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSampling(Nullable<Int32>)` | Initializes a new instance of the RandomSampling class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `UseBatchDiversity` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeInformativenessScores(IFullModel<,Tensor<>,Tensor<>>,Tensor<>)` |  |
| `GetSelectionStatistics` |  |
| `SelectSamples(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,Int32)` |  |
| `UpdateStatistics(Vector<>)` | Updates selection statistics. |

