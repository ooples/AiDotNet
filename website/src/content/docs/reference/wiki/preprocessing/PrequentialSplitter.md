---
title: "PrequentialSplitter<T>"
description: "Prequential (predictive sequential) evaluation splitter for online/streaming data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Online`

Prequential (predictive sequential) evaluation splitter for online/streaming data.

## For Beginners

Prequential evaluation is the standard way to evaluate models
on streaming data. For each new sample: first use it as a test sample (predict its label),
then add it to the training set. This simulates real-world continuous learning.

## How It Works

**How It Works:**

1. Start with an initial training window
2. For each subsequent sample:
- Test: Predict using current model
- Train: Update model with the true label
3. Evaluation is "test-then-train" on every sample

**When to Use:**

- Streaming/online learning scenarios
- Concept drift detection
- Real-time prediction systems
- Evaluating adaptive algorithms

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrequentialSplitter(Int32,Int32,Boolean,Int32,Int32)` | Creates a new prequential evaluation splitter. |

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

