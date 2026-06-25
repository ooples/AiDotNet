---
title: "IterativeStratificationSplitter<T>"
description: "Iterative stratification splitter for multi-label classification problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified`

Iterative stratification splitter for multi-label classification problems.

## For Beginners

Standard stratification works well for single-label classification
where each sample belongs to exactly one class. But what about multi-label problems
where each sample can have multiple labels (like a movie being both "Action" AND "Comedy")?

## How It Works

**How It Works:**

1. Calculate label frequencies across all samples
2. Start with rarest label combinations
3. Iteratively assign samples to folds, prioritizing balance for rare labels
4. Continue until all samples are assigned

**Example:**
Document classification where each document can have multiple topics:

- Doc1: [Politics, Economy]
- Doc2: [Sports]
- Doc3: [Politics, Sports]
- Doc4: [Economy]

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IterativeStratificationSplitter(Double,Matrix<>,Boolean,Int32)` | Creates a new Iterative Stratification splitter for multi-label data. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `WithLabelMatrix(Matrix<>)` | Sets the multi-label matrix. |

