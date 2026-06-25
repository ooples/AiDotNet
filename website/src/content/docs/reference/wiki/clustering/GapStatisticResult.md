---
title: "GapStatisticResult"
description: "Results from Gap Statistic analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Results from Gap Statistic analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `GapValues` | Gap values for each K. |
| `KValues` | The K values tested. |
| `OptimalK` | The optimal number of clusters based on the Gap criterion. |
| `ReferenceWCSSValues` | Reference (expected) WCSS values for each K. |
| `StandardErrors` | Standard errors for each Gap value. |
| `WCSSValues` | Within-cluster sum of squares for each K. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary of the Gap analysis. |

