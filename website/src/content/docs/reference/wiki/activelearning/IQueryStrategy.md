---
title: "IQueryStrategy<T, TInput, TOutput>"
description: "Interface for query strategies in active learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for query strategies in active learning.

## For Beginners

A query strategy determines which unlabeled samples
should be selected for labeling by the oracle (human expert). Different strategies
use different criteria for selection.

## How It Works

**Common Query Strategies:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of how the strategy works. |
| `Name` | Gets the name of the query strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` | Computes informativeness scores for all samples in the unlabeled pool. |
| `Reset` | Resets the strategy to its initial state. |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` | Selects the most informative samples from the unlabeled pool. |
| `UpdateState(Int32[],[])` | Updates the strategy's internal state after new samples are labeled. |

