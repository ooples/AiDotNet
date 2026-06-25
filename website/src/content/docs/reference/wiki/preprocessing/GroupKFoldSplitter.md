---
title: "GroupKFoldSplitter<T>"
description: "K-Fold cross-validation that keeps samples from the same group together."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

K-Fold cross-validation that keeps samples from the same group together.

## For Beginners

Sometimes your data has natural groups that should stay together.
For example, if you have multiple measurements from the same patient, you don't want
patient A's Monday measurement in training and their Tuesday measurement in test -
that would be data leakage!

## How It Works

**How It Works:**
Instead of splitting by samples, we split by groups:

**Common Use Cases:**

- Medical studies: Multiple measurements per patient
- User studies: Multiple sessions per user
- Multi-site studies: Multiple samples per location

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupKFoldSplitter(Int32[],Nullable<Int32>,Boolean,Int32)` | Creates a new Group K-Fold splitter. |

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

