---
title: "CoresetSelectorOptions"
description: "Configuration options for coreset selection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for coreset selection.

## How It Works

Coreset selection finds a small representative subset of the data that approximates
the full dataset for training, reducing compute while preserving model quality.

## Properties

| Property | Summary |
|:-----|:--------|
| `Seed` | Random seed for reproducibility. |
| `SelectionRatio` | Target size of the coreset as a fraction of the original dataset. |
| `Strategy` | Strategy for coreset selection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

