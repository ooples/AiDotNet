---
title: "BenchmarkDataSelectionSummary"
description: "Summarizes how data was selected/sampled for a dataset-backed benchmark suite run."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Benchmarking.Models`

Summarizes how data was selected/sampled for a dataset-backed benchmark suite run.

## Properties

| Property | Summary |
|:-----|:--------|
| `CiMode` | Gets whether CI mode was enabled for this run. |
| `ClientsUsed` | Gets the number of clients/users included. |
| `FeatureCount` | Gets the feature count for the aggregated dataset. |
| `MaxSamplesPerUser` | Gets the maximum samples per user applied (0 means not applied). |
| `Seed` | Gets the seed used for deterministic sampling when applicable. |
| `TestSamplesUsed` | Gets the number of aggregated test samples included. |
| `TrainSamplesUsed` | Gets the number of aggregated training samples included. |

