---
title: "AdversarialValidationSplitter<T>"
description: "Adversarial validation splitter that identifies samples most similar to test distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Adversarial validation splitter that identifies samples most similar to test distribution.

## For Beginners

Sometimes your training and test data come from different distributions
(e.g., train from 2022, test from 2023). Adversarial validation helps detect this problem.

## How It Works

**How It Works:**

1. Train a classifier to distinguish train vs test samples
2. If the classifier performs well (AUC > 0.5), there's distribution shift
3. Use the classifier's predictions to create a more realistic validation set
4. Put samples most "test-like" into validation

**Practical Use:**
This splitter doesn't actually train a classifier - it creates indices that you can use
after running adversarial validation externally. You provide the probability scores
from your adversarial classifier.

**When to Use:**

- Kaggle competitions where public/private test differs from train
- Time-based production scenarios
- Any situation with potential train/test distribution shift

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialValidationSplitter(Double[],Double,Int32)` | Creates an adversarial validation splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

