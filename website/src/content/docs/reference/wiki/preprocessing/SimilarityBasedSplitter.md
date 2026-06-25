---
title: "SimilarityBasedSplitter<T>"
description: "Similarity-based splitter that splits data based on sample similarity scores."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Similarity-based splitter that splits data based on sample similarity scores.

## For Beginners

This splitter uses similarity between samples to create splits.
Samples that are very similar to training data are placed in the test set to evaluate
interpolation, while dissimilar samples test extrapolation capability.

## How It Works

**Modes:**

- Interpolation Test: Test samples similar to training (tests within-distribution generalization)
- Extrapolation Test: Test samples dissimilar to training (tests out-of-distribution performance)

**When to Use:**

- When you want to specifically test interpolation vs extrapolation
- For robustness evaluation
- When similarity structure in data is meaningful

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimilarityBasedSplitter(Double,Double,Boolean,Boolean,Int32)` | Creates a new similarity-based splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

