---
title: "BenchmarkMetric"
description: "Defines standardized metrics used in benchmark reports."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines standardized metrics used in benchmark reports.

## For Beginners

Benchmarks produce scores and measurements. This enum provides a type-safe,
standardized vocabulary for common metrics so we avoid "stringly-typed" metric keys.

## Fields

| Field | Summary |
|:-----|:--------|
| `Accuracy` | Proportion of correct answers (0.0 to 1.0). |
| `AverageConfidence` | Average confidence value (0.0 to 1.0) when available. |
| `AverageTimePerItemMilliseconds` | Average time per item in milliseconds. |
| `CorrectCount` | Number of correct items. |
| `MeanSquaredError` | Mean squared error (regression-style error metric). |
| `RootMeanSquaredError` | Root mean squared error (regression-style error metric). |
| `TotalDurationMilliseconds` | Total duration in milliseconds. |
| `TotalEvaluated` | Total number of items evaluated. |

