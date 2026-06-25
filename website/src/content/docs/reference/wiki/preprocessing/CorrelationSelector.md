---
title: "CorrelationSelector<T>"
description: "Feature selector that removes highly correlated features to reduce redundancy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation`

Feature selector that removes highly correlated features to reduce redundancy.

## For Beginners

This selector removes duplicate or redundant information from your data.

Imagine you're collecting data about houses and include both:

- Square footage of the house
- Number of rooms
- Price

Square footage and number of rooms are often highly correlated (bigger houses tend to have
more rooms). This selector would detect this relationship and might keep only one of these
features, reducing redundancy while preserving the most important information.

By eliminating redundant features:

- Your model trains faster
- You reduce the risk of overfitting
- The model becomes easier to interpret and explain

The threshold setting controls how strict this filtering is:

- Higher values (e.g., 0.9) allow more features to be included
- Lower values (e.g., 0.5) result in more features being removed

## How It Works

CorrelationSelector identifies and removes features that are highly correlated with other
features. When features are strongly correlated, they provide redundant information, so
keeping just one of them can simplify your model without losing predictive power.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CorrelationSelector(Double,Int32[])` | Creates a new instance of `CorrelationSelector`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CorrelationMatrix` | Gets the correlation matrix computed during fitting. |
| `SelectedCount` | Gets the number of selected features after fitting. |
| `SelectedFeatures` | Gets the indices of selected features after fitting. |
| `SupportsInverseTransform` | Gets whether this transformer supports inverse transformation. |
| `Threshold` | Gets the correlation threshold above which features are considered highly correlated. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCorrelationMatrix(Matrix<>)` | Computes the Pearson correlation matrix for all features. |
| `FitCore(Matrix<>)` | Computes the correlation matrix and determines which features to keep. |
| `GetCorrelationsWithSelected(Int32)` | Gets the correlations between a specific feature and all selected features. |
| `GetFeatureNamesOut(String[])` | Gets the output feature names after transformation. |
| `GetSupportMask` | Gets a boolean mask indicating which features are selected. |
| `InverseTransformCore(Matrix<>)` | Inverse transformation is not supported for feature selection. |
| `TransformCore(Matrix<>)` | Removes features that are highly correlated with other selected features. |

