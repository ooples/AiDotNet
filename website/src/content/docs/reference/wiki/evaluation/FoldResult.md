---
title: "FoldResult<T>"
description: "Results from a single cross-validation fold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Results from a single cross-validation fold.

## Properties

| Property | Summary |
|:-----|:--------|
| `Actuals` | Actual values from the validation set. |
| `FoldIndex` | The fold index (0-based). |
| `Metrics` | Metrics computed on this fold's validation set. |
| `Predictions` | Predictions on the validation set. |
| `TrainSize` | Number of training samples in this fold. |
| `ValidationSize` | Number of validation samples in this fold. |

