---
title: "RankedBatchStrategy<T, TInput, TOutput>"
description: "Simple ranked batch selection strategy with diversity filtering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Batch`

Simple ranked batch selection strategy with diversity filtering.

## For Beginners

This is the simplest batch selection strategy.
It ranks samples by their informativeness scores and selects the top-k,
optionally filtering out samples that are too similar to already-selected ones.

## How It Works

**How It Works:**

**Trade-offs:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RankedBatchStrategy` | Initializes a new RankedBatchStrategy with default settings. |
| `RankedBatchStrategy(Double,Double)` | Initializes a new RankedBatchStrategy with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityTradeoff` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiversity(,)` |  |
| `SelectBatch(Int32[],Vector<>,IDataset<,,>,Int32)` |  |

