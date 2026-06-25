---
title: "OversamplingBase<T>"
description: "Base class for oversampling strategies that create synthetic samples for minority classes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Preprocessing.ImbalancedLearning`

Base class for oversampling strategies that create synthetic samples for minority classes.

## For Beginners

When you have imbalanced data (e.g., 1000 "normal" samples but only
50 "fraud" samples), the model often ignores the minority class. Oversampling creates
synthetic "fraud" samples so the model sees enough examples to learn the pattern.

Different oversampling strategies create synthetic samples in different ways:

- SMOTE: Creates samples between existing minority samples
- ADASYN: Creates more samples in regions where classification is harder
- BorderlineSMOTE: Focuses on samples near the decision boundary

Important: Only oversample the training data! Never oversample test data.

## How It Works

Oversampling strategies increase the number of minority class samples by creating synthetic
examples. This helps machine learning models learn to recognize minority classes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OversamplingBase(Double,Int32,Nullable<Int32>)` | Initializes a new instance of the OversamplingBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this oversampling strategy. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EuclideanDistance(Vector<>,Vector<>)` | Computes the Euclidean distance between two vectors. |
| `FindKNearestNeighbors(Matrix<>,Int32,List<Int32>,Int32)` | Finds the k nearest neighbors of a sample within a set of candidates. |
| `GenerateSyntheticSamples(Matrix<>,List<Int32>,Int32)` | Generates synthetic samples for a minority class. |
| `GetClassCounts(Vector<>)` | Gets the count of samples per class. |
| `GetClassIndices(Vector<>,)` | Gets the indices of samples belonging to a specific class. |
| `GetStatistics` | Gets statistics about the last resampling operation. |
| `Resample(Matrix<>,Vector<>)` | Resamples the dataset by creating synthetic minority samples. |

## Fields

| Field | Summary |
|:-----|:--------|
| `KNeighbors` | Number of nearest neighbors to use. |
| `LastStatistics` | Statistics about the last resampling operation. |
| `NumOps` | Numeric operations helper for generic math. |
| `Random` | Random number generator. |
| `SamplingStrategy` | The target ratio of minority to majority class samples. |

