---
title: "DataPrunerOptions"
description: "Configuration options for data pruning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for data pruning.

## How It Works

Data pruning removes easy or redundant samples based on training signals
(e.g., loss, confidence, forgetting events), keeping only the most informative examples.

## Properties

| Property | Summary |
|:-----|:--------|
| `MinEpochsForScoring` | Minimum number of epochs before pruning scores are reliable. |
| `PruneRatio` | Fraction of data to prune (remove). |
| `Strategy` | Strategy for selecting samples to prune. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

