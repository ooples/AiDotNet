---
title: "MissingIndicator<T>"
description: "Creates binary indicator features for missing values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Imputers`

Creates binary indicator features for missing values.

## For Beginners

Sometimes knowing that data is missing is important:

- A missing income might mean someone declined to answer (high income?)
- A missing medical test might mean the doctor didn't think it was necessary

This transformer adds new columns (one per feature) with 1 where data was missing
and 0 where it was present.

## How It Works

MissingIndicator transforms a dataset by adding binary columns that indicate
where values were missing. This is useful when the fact that a value is missing
is itself informative for the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MissingIndicator(MissingIndicatorFeatures,Double,Int32[])` | Creates a new instance of `MissingIndicator`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Features` | Gets which features to create indicators for. |
| `FeaturesWithMissing` | Gets the indices of features that had missing values during fit. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FitCore(Matrix<>)` | Identifies which features have missing values. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported. |
| `TransformCore(Matrix<>)` | Creates binary indicator features for missing values. |

