---
title: "NemenyiResult<T>"
description: "Results from Nemenyi post-hoc test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Results from Nemenyi post-hoc test.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Significance level used. |
| `AverageRanks` | Average ranks for each algorithm. |
| `CriticalDifference` | Critical difference for significance at the specified alpha level. |
| `NumAlgorithms` | Number of algorithms compared. |
| `NumDatasets` | Number of datasets used. |
| `SignificantDifferences` | Matrix of significant differences: true if algorithms i and j differ significantly. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSignificantPairs` | Gets pairs of algorithms that are significantly different. |
| `ToString` | Returns a summary of the test results. |

