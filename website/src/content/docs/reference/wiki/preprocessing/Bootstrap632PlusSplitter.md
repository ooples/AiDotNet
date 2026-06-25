---
title: "Bootstrap632PlusSplitter<T>"
description: ".632+ Bootstrap splitter that improves upon .632 Bootstrap for high-variance scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Bootstrap`

.632+ Bootstrap splitter that improves upon .632 Bootstrap for high-variance scenarios.

## For Beginners

The .632 bootstrap can still be biased when the model overfits.
The .632+ method adaptively adjusts the weighting based on how much the model overfits,
providing more reliable estimates.

## How It Works

**How It Works:**
The .632+ error is calculated as:
error_632+ = (1-w)*error_train + w*error_test
where w varies between 0.632 and 1 based on overfitting degree.

**Mathematical Details:**

- R = (error_test - error_train) / (gamma - error_train)
- gamma = error under the null model (random predictions)
- w = 0.632 / (1 - 0.368 * R)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Bootstrap632PlusSplitter(Int32,Int32)` | Creates a new .632+ Bootstrap splitter. |

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

