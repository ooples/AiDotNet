---
title: "TomekLinks<T>"
description: "Implements Tomek Links undersampling for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements Tomek Links undersampling for handling imbalanced datasets.

## For Beginners

A Tomek link is a special relationship between two samples:

1. Sample A (minority class) and Sample B (majority class)
2. A's nearest neighbor is B
3. B's nearest neighbor is A
4. They are each other's closest sample across classes!

Why Tomek links are important:

- They represent borderline or noisy samples
- Removing the majority sample from a Tomek link cleans the decision boundary
- It helps the classifier focus on clearer cases

Visual example:
```
Before: M . . . . m M . . . m . . . . M
^ ^
These two might be a Tomek link

After: . . . . . m . . . . m . . . . M
Majority sample removed, cleaner boundary
```

M = majority, m = minority

When to use:

- Data cleaning before training
- In combination with oversampling (SMOTE + Tomek)
- When you want minimal but targeted undersampling

References:

- Tomek (1976). "Two Modifications of CNN"

## How It Works

Tomek Links removes majority class samples that form "Tomek links" with minority
samples. A Tomek link is a pair of samples from different classes that are each
other's nearest neighbor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TomekLinks` | Initializes a new instance of the TomekLinks class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this undersampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindNearestNeighbor(Matrix<>,Int32)` | Finds the nearest neighbor of a sample. |
| `GetClassCounts(Vector<>)` | Gets the count of samples per class. |
| `GetClassIndices(Vector<>,)` | Gets the indices of samples belonging to a specific class. |
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset by removing Tomek links. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for generic math. |
| `_lastStatistics` | Statistics about the last resampling operation. |

