---
title: "QueryByCommitteeSplitter<T>"
description: "Query-by-Committee (QBC) splitter for ensemble-based active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning`

Query-by-Committee (QBC) splitter for ensemble-based active learning.

## For Beginners

Query-by-Committee maintains multiple models (a "committee")
that are trained on different subsets of the labeled data. Samples where the committee
disagrees most are the most informative and should be labeled next.

## How It Works

**How It Works:**

1. Create multiple subsets of the initial labeled data (one per committee member)
2. Each subset can be used to train a different model
3. Unlabeled samples with highest disagreement are prioritized for labeling

**When to Use:**

- When you can train multiple models
- Ensemble-based approaches
- Maximizing information gain from labeling
- Reducing labeling costs through disagreement sampling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryByCommitteeSplitter(Int32,Double,Double,Boolean,Int32)` | Creates a new Query-by-Committee splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

