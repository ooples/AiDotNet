---
title: "CovariateShiftSplitter<T>"
description: "Covariate shift splitter that creates intentional distribution shift between train and test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized`

Covariate shift splitter that creates intentional distribution shift between train and test.

## For Beginners

Covariate shift occurs when the distribution of input features
differs between training and test data, even though the relationship between inputs
and outputs remains the same. This splitter intentionally creates such a shift.

## How It Works

**How It Works:**

1. Identify a primary feature dimension for the shift
2. Select training samples from one range of that feature
3. Select test samples from a different range

**When to Use:**

- Testing model robustness to distribution shift
- Simulating real-world deployment scenarios
- Evaluating domain adaptation techniques

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CovariateShiftSplitter(Double,Double,Int32,Int32)` | Creates a new covariate shift splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

