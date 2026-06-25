---
title: "ProfileComparison"
description: "Result of comparing two profiling reports."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diagnostics`

Result of comparing two profiling reports.

## Properties

| Property | Summary |
|:-----|:--------|
| `Baseline` | Baseline report. |
| `Current` | Current report being compared. |
| `HasRegressions` | Whether any regressions were detected. |
| `Improvements` | Operations that improved beyond threshold. |
| `Regressions` | Operations that regressed beyond threshold. |
| `ThresholdPercent` | Threshold percentage used for comparison. |

