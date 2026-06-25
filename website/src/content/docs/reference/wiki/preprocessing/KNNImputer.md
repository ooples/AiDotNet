---
title: "KNNImputer<T>"
description: "Imputes missing values using K-Nearest Neighbors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Imputers`

Imputes missing values using K-Nearest Neighbors.

## For Beginners

This imputer fills in missing values by looking at similar data points:

- Finds the K most similar rows that have the value you need
- Uses their average to fill in the missing value
- "Similar" is measured using Euclidean distance on non-missing features

Example: If you're missing someone's income, KNN finds K similar people
(same age, education, etc.) and uses their average income.

## How It Works

KNNImputer replaces missing values with the mean (or weighted mean) of the K nearest
neighbors found in the training set. Each sample's missing values are imputed using
the values from the K most similar samples that have non-missing values for that feature.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNNImputer(Int32,KNNWeights,Double,Int32[])` | Creates a new instance of `KNNImputer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NNeighbors` | Gets the number of neighbors to use for imputation. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Weights` | Gets the weighting scheme used for neighbors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Stores the training data for neighbor lookup. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for KNN imputation. |
| `TransformCore(Matrix<>)` | Imputes missing values using K-nearest neighbors. |

