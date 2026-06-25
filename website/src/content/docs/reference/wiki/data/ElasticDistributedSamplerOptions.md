---
title: "ElasticDistributedSamplerOptions"
description: "Configuration options for elastic distributed sampling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Sampling`

Configuration options for elastic distributed sampling.

## Properties

| Property | Summary |
|:-----|:--------|
| `DatasetSize` | Total number of samples in the dataset. |
| `DropLast` | Whether to drop the remainder so all replicas get exactly DatasetSize/NumReplicas samples. |
| `NumReplicas` | Number of distributed workers (replicas). |
| `Rank` | Rank of the current worker (0-based). |
| `Seed` | Random seed for reproducibility. |
| `Shuffle` | Whether to shuffle indices each epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

