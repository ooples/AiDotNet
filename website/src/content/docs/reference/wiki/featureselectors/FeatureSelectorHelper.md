---
title: "FeatureSelectorHelper<T, TInput>"
description: "Provides common helper methods for feature selection algorithms."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.FeatureSelectors`

Provides common helper methods for feature selection algorithms.

## For Beginners

This class contains shared functionality that different feature selection methods
can use. It helps avoid duplicating code and keeps the feature selectors more focused on their
specific selection strategies rather than common data handling tasks.

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateMaxFeature(Tensor<>,Int32,Int32)` | Finds the maximum value across all dimensions for a specific feature and sample. |
| `CalculateMaxRecursive(Tensor<>,Int32[],Int32,,Boolean)` | Recursively traverses tensor dimensions to find the maximum value. |
| `CalculateMeanFeature(Tensor<>,Int32,Int32)` | Calculates the mean value across all dimensions for a specific feature and sample. |
| `CalculateMeanRecursive(Tensor<>,Int32[],Int32,,Int32)` | Recursively traverses tensor dimensions to calculate the sum and count of elements. |
| `CalculateWeightedSum(Tensor<>,Int32,Int32,Dictionary<Int32,>)` | Calculates a weighted sum of values across all dimensions for a specific feature and sample. |
| `CalculateWeightedSumRecursive(Tensor<>,Int32[],Int32,Dictionary<Int32,>,)` | Recursively traverses tensor dimensions to calculate a weighted sum of elements. |
| `CopyFeature(Tensor<>,Tensor<>,Int32,Int32,Int32)` | Copies a feature from the source tensor to the destination tensor. |
| `CreateFeatureSubset(,List<Int32>)` | Creates a subset of features from the original input data. |
| `CreateFilteredData(,List<Int32>)` | Creates a new data structure containing only the selected features. |
| `ExtractFeatureVector(,Int32,Int32,FeatureExtractionStrategy,Dictionary<Int32,>)` | Extracts a feature vector from the input data. |
| `GetFirstElement(Tensor<>,Int32,Int32)` | Gets the first element for a specific feature and sample. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations for numeric calculations with type T. |

