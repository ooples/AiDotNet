---
title: "BenchmarkSuiteReport"
description: "Represents the outcome and metrics for a single benchmark suite run."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Benchmarking.Models`

Represents the outcome and metrics for a single benchmark suite run.

## Properties

| Property | Summary |
|:-----|:--------|
| `CategoryAccuracies` | Gets optional category-level accuracy breakdowns (when available and requested). |
| `DataSelection` | Gets optional dataset selection details for dataset-backed suites. |
| `Duration` | Gets the duration for this suite. |
| `EndedUtc` | Gets the UTC time when this suite ended. |
| `FailureReason` | Gets an optional failure reason when `Status` is `Failed`. |
| `Kind` | Gets the suite kind/category. |
| `Metrics` | Gets the standardized metrics for this suite. |
| `Name` | Gets the suite display name. |
| `StartedUtc` | Gets the UTC time when this suite started. |
| `Status` | Gets the execution status. |
| `Suite` | Gets the benchmark suite identifier. |

