---
title: "SubmodularBatchStrategy<T, TInput, TOutput>"
description: "Submodular batch selection strategy using facility location objectives."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Batch`

Submodular batch selection strategy using facility location objectives.

## For Beginners

Submodular functions have a special property called
"diminishing returns" - adding more similar samples provides less and less benefit.
This naturally encourages selecting diverse, representative samples.

## How It Works

**How It Works:**

**Facility Location Objective:**

F(S) = Σ max_{s∈S} sim(x, s) for all x in the data

This objective ensures selected samples cover the entire data space well.

**Advantages:**

**Reference:** Wei et al. "Submodularity in Data Subset Selection and Active Learning" (ICML 2015)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SubmodularBatchStrategy` | Initializes a new SubmodularBatchStrategy with default settings. |
| `SubmodularBatchStrategy(SubmodularObjective,Double)` | Initializes a new SubmodularBatchStrategy with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityTradeoff` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiversity(,)` |  |
| `ComputeMarginalGain(IReadOnlyList<Int32>,Int32,IDataset<,,>)` |  |
| `GreedyMaximization(Int32[],IDataset<,,>,Int32)` |  |
| `SelectBatch(Int32[],Vector<>,IDataset<,,>,Int32)` |  |

