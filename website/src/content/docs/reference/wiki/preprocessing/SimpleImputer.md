---
title: "SimpleImputer<T>"
description: "Imputes missing values using simple strategies like mean, median, or constant."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Imputers`

Imputes missing values using simple strategies like mean, median, or constant.

## For Beginners

This transformer fills in gaps in your data:

- If you have missing ages, replace them with average age (Mean)
- If you have missing incomes with outliers, use median income (Median)
- If you have missing categories, use most common category (MostFrequent)
- Or fill with a specific value like 0 or -1 (Constant)

Example with Mean strategy:
[1, 2, NaN, 4, 5] → [1, 2, 3, 4, 5] (NaN replaced with mean=3)

## How It Works

SimpleImputer fills in missing values (represented as NaN) using simple strategies:

- Mean: Replace with column mean
- Median: Replace with column median
- MostFrequent: Replace with most common value
- Constant: Replace with a specified value

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimpleImputer(ImputationStrategy,,Int32[])` | Creates a new instance of `SimpleImputer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FillValue` | Gets the fill value used for Constant strategy. |
| `MissingValue` | Gets the value considered as missing. |
| `Statistics` | Gets the computed statistics for each feature. |
| `Strategy` | Gets the imputation strategy used. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Computes the statistics for each feature from the training data. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for imputation. |
| `TransformCore(Matrix<>)` | Transforms the data by imputing missing values. |

