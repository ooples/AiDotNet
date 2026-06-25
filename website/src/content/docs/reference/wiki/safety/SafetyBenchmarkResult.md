---
title: "SafetyBenchmarkResult"
description: "Results from running a safety benchmark."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Results from running a safety benchmark.

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageLatencyMs` | Average evaluation latency in milliseconds. |
| `CategoryResults` | Per-category benchmark results. |
| `F1Score` | F1 Score: Harmonic mean of precision and recall. |
| `FalseNegatives` | Number of unsafe content missed. |
| `FalsePositiveRate` | False Positive Rate: FP / (FP + TN). |
| `FalsePositives` | Number of safe content incorrectly flagged. |
| `Precision` | Precision: TP / (TP + FP). |
| `Recall` | Recall: TP / (TP + FN). |
| `TotalTestCases` | Total number of test cases evaluated. |
| `TrueNegatives` | Number of correctly identified safe content. |
| `TruePositives` | Number of correctly identified unsafe content. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Empty` | An empty benchmark result. |

