---
title: "ActiveLearningQueryStrategyOptions"
description: "Configuration options for active learning query strategies."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Quality`

Configuration options for active learning query strategies.

## How It Works

Active learning selects the most informative unlabeled samples for annotation,
maximizing model improvement per labeling dollar spent.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumMcDropoutPasses` | Number of Monte Carlo dropout forward passes for BALD. |
| `QueryBatchSize` | Number of samples to select per query round. |
| `Seed` | Random seed for reproducibility. |
| `Strategy` | Query strategy for sample selection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

