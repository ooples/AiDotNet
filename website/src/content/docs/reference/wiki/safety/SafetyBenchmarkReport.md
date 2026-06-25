---
title: "SafetyBenchmarkReport"
description: "Comprehensive report from running all safety benchmarks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Benchmarking`

Comprehensive report from running all safety benchmarks.

## For Beginners

SafetyBenchmarkReport provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `AggregateScore` | Overall aggregate score across all benchmarks (0.0-1.0). |
| `BenchmarkResults` | Individual benchmark results by benchmark name. |
| `OverallF1` | Overall F1 score across all benchmarks. |
| `OverallPrecision` | Overall precision across all benchmarks. |
| `OverallRecall` | Overall recall across all benchmarks. |
| `Recommendations` | Recommendations based on benchmark results. |
| `TotalTestCases` | Total test cases across all benchmarks. |

