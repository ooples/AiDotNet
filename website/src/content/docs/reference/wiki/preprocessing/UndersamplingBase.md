---
title: "UndersamplingBase<T>"
description: "Base class for undersampling strategies that reduce majority class samples."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing.ImbalancedLearning`

Base class for undersampling strategies that reduce majority class samples.

## For Beginners

If you have 1000 "normal" samples and 50 "fraud" samples,
undersampling might reduce the "normal" samples to 50-100 to balance the classes.

Advantages of undersampling:

- Reduces training time (fewer samples)
- Simpler than synthetic generation
- Works well with large datasets

Disadvantages:

- Loses potentially useful information
- Can underfit if too aggressive
- Not suitable for small datasets

Different undersampling strategies choose which samples to remove:

- Random: Remove majority samples randomly
- Tomek Links: Remove majority samples that form Tomek links with minority
- ENN: Remove samples misclassified by nearest neighbors
- NearMiss: Keep majority samples closest to minority samples

## How It Works

Undersampling strategies reduce the number of majority class samples to achieve
a more balanced dataset. Unlike oversampling, no synthetic samples are created.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UndersamplingBase(Double,Nullable<Int32>)` | Initializes a new instance of the UndersamplingBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this undersampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EuclideanDistance(Vector<>,Vector<>)` | Computes the Euclidean distance between two vectors. |
| `GetClassCounts(Vector<>)` | Gets the count of samples per class. |
| `GetClassIndices(Vector<>,)` | Gets the indices of samples belonging to a specific class. |
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset by removing majority samples. |
| `SelectSamplesToKeep(Matrix<>,Vector<>,List<Int32>,List<Int32>,Int32)` | Selects which majority samples to keep. |

## Fields

| Field | Summary |
|:-----|:--------|
| `LastStatistics` | Statistics about the last resampling operation. |
| `NumOps` | Numeric operations helper for generic math. |
| `Random` | Random number generator. |
| `SamplingStrategy` | The target ratio of minority to majority class samples. |

