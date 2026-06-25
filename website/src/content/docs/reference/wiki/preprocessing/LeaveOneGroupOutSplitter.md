---
title: "LeaveOneGroupOutSplitter<T>"
description: "Leave-One-Group-Out cross-validation where each group is the test set once."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.GroupBased`

Leave-One-Group-Out cross-validation where each group is the test set once.

## For Beginners

This is like Leave-One-Out, but for groups instead of samples.
Each unique group becomes the test set while all other groups form the training set.

## How It Works

**Example - Patient Study:**

**When to Use:**

- Cross-subject validation in user studies
- Cross-patient validation in medical research
- When you want to test generalization to new groups

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeaveOneGroupOutSplitter(Int32[])` | Creates a new Leave-One-Group-Out splitter. |

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

