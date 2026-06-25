---
title: "MultiSourceMixerOptions"
description: "Configuration options for multi-source data mixing."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Pipeline`

Configuration options for multi-source data mixing.

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferSize` | Buffer size for interleaving. |
| `Seed` | Random seed for reproducibility. |
| `StopOnShortestSource` | Whether to stop when the smallest source is exhausted. |
| `Weights` | Mixing weights for each data source (normalized internally). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

