---
title: "EditedNearestNeighbors<T>"
description: "Implements Edited Nearest Neighbors (ENN) undersampling for handling imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.ImbalancedLearning`

Implements Edited Nearest Neighbors (ENN) undersampling for handling imbalanced datasets.

## For Beginners

ENN uses a simple rule to decide which samples to remove:

"If a sample's neighbors mostly disagree with its label, remove it."

For example, if a majority class sample has 3 nearest neighbors:

- 2 are minority class
- 1 is majority class

Then the sample is "misclassified" by its neighbors and is removed.

This is like asking: "Would a K-NN classifier with K=3 correctly classify this sample?"
If no, the sample is probably noisy or on the wrong side of the boundary.

Visual example:
```
Before: M M M m M m m m m
^
This M surrounded by m's would be removed

After: M M M m m m m m
Cleaner boundary
```

When to use:

- Data cleaning before training
- In combination with SMOTE (SMOTE + ENN = SMOTEENN)
- When you want to remove ambiguous/noisy samples

References:

- Wilson (1972). "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data"

## How It Works

ENN removes samples whose class label differs from the majority of their k nearest
neighbors. This removes noisy and borderline samples from the majority class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EditedNearestNeighbors(Int32,ENNKind)` | Initializes a new instance of the EditedNearestNeighbors class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this undersampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FindKNearestNeighbors(Matrix<>,Int32,List<Int32>,Int32)` | Finds the k nearest neighbors of a sample. |
| `GetClassCounts(Vector<>)` | Gets the count of samples per class. |
| `GetClassIndices(Vector<>,)` | Gets the indices of samples belonging to a specific class. |
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset by removing samples misclassified by their neighbors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations helper for generic math. |
| `_kNeighbors` | Number of nearest neighbors to consider. |
| `_kind` | The editing kind to use. |
| `_lastStatistics` | Statistics about the last resampling operation. |

