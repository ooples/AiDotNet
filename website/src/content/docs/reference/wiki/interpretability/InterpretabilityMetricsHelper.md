---
title: "InterpretabilityMetricsHelper<T>"
description: "Provides static utility methods for computing interpretability and fairness metrics."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Interpretability`

Provides static utility methods for computing interpretability and fairness metrics.

## For Beginners

This is a collection of reusable helper methods for fairness and bias analysis.

These methods handle common tasks like:

- Identifying unique groups in data (e.g., different age groups, genders)
- Computing metrics like positive rates, true positive rates, etc.
- Extracting subsets of data for specific groups

By centralizing these methods here, we avoid code duplication and ensure consistent
calculations across all bias detection and fairness evaluation tools.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFalsePositiveRate(Vector<>,Vector<>)` | Computes the False Positive Rate (FPR). |
| `ComputePositiveRate(Vector<>)` | Computes the positive prediction rate (proportion of positive predictions). |
| `ComputePrecision(Vector<>,Vector<>)` | Computes the Precision (Positive Predictive Value). |
| `ComputeTruePositiveRate(Vector<>,Vector<>)` | Computes the True Positive Rate (TPR) or Recall. |
| `GetGroupIndices(Vector<>,)` | Gets the indices of all samples belonging to a specific group. |
| `GetSubset(Vector<>,List<Int32>)` | Extracts a subset of a vector based on specified indices. |
| `GetUniqueGroups(Vector<>)` | Identifies all unique groups in the sensitive feature. |

