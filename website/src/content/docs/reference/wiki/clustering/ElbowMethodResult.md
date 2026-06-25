---
title: "ElbowMethodResult"
description: "Results from Elbow Method analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Evaluation`

Results from Elbow Method analysis.

## Properties

| Property | Summary |
|:-----|:--------|
| `ElbowK` | The detected elbow point (optimal K). |
| `ImprovementRates` | Improvement rate from K-1 to K for each K. |
| `KValues` | The K values tested. |
| `WCSSValues` | Within-cluster sum of squares for each K. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary of the Elbow analysis. |

