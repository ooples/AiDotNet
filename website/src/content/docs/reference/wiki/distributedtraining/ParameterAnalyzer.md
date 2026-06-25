---
title: "ParameterAnalyzer<T>"
description: "Analyzes model parameters and creates optimized groupings for distributed communication."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Analyzes model parameters and creates optimized groupings for distributed communication.

## For Beginners

Think of ParameterAnalyzer as a smart packing assistant. When shipping items, you don't
want to send thousands of tiny packages - it's inefficient! Instead, you group small
items together into larger boxes.

## How It Works

Similarly, when communicating parameters across GPUs:

- Sending many small parameter arrays is slow (lots of communication overhead)
- Grouping small parameters together reduces the number of messages
- This analyzer figures out the best way to group parameters for efficiency

For example, instead of sending 1000 separate bias vectors (each with 1 parameter),
we might group them into 10 larger chunks (each with 100 parameters).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParameterAnalyzer(Int32,Int32)` | Creates a new parameter analyzer with the specified settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeForDistribution(Vector<>)` | Analyzes parameters and creates groups optimized for even distribution across processes. |
| `AnalyzeModel(IFullModel<,,>)` | Analyzes a model's parameters and creates optimized groupings. |
| `AnalyzeParameters(Vector<>)` | Analyzes a parameter vector and creates optimized groupings. |
| `CalculateDistributionStats(List<ParameterAnalyzer<>.ParameterGroup>)` | Calculates statistics about parameter distribution for a model. |
| `ValidateGrouping(List<ParameterAnalyzer<>.ParameterGroup>,Int32)` | Validates that parameter groups cover all parameters without gaps or overlaps. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DISTRIBUTION_GROUP_DIVISOR` | Divisor for calculating base group size for distributed training. |
| `REMAINING_PARAMS_MERGE_THRESHOLD` | Threshold multiplier for merging remaining parameters into the last group. |
| `SMALL_GROUP_MERGE_DIVISOR` | Threshold divisor for merging small final groups. |

