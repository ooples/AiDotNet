---
title: "RandomOverSampler<T>"
description: "Implements random oversampling for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements random oversampling for handling imbalanced datasets.

## For Beginners

If you have 1000 "normal" samples and 100 "fraud" samples,
random oversampling randomly duplicates "fraud" samples until you have enough.

Advantages:

- Very simple and fast
- Good baseline to compare against
- Preserves original data characteristics

Disadvantages:

- Creates exact duplicates (no diversity)
- Can lead to overfitting on duplicated samples
- Model may "memorize" rather than "learn"

Comparison with SMOTE:

- RandomOverSampler: Duplicates existing samples
- SMOTE: Creates new synthetic samples between existing ones
- SMOTE usually performs better, but RandomOverSampler is simpler

When to use:

- As a baseline for comparison
- When synthetic samples might introduce artifacts
- When you have very few minority samples (SMOTE needs at least k+1)

References:

- Kotsiantis et al. (2006). "Handling imbalanced datasets: A review"

## How It Works

Random oversampling duplicates random samples from the minority class until
the desired balance is achieved. It's the simplest oversampling method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomOverSampler(Double,Nullable<Int32>)` | Initializes a new instance of the RandomOverSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this oversampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateSyntheticSamples(Matrix<>,List<Int32>,Int32)` | Generates synthetic samples by duplicating existing minority samples. |

