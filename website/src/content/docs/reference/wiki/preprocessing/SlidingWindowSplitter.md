---
title: "SlidingWindowSplitter<T>"
description: "Sliding window splitter with fixed-size training window that moves through time."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries`

Sliding window splitter with fixed-size training window that moves through time.

## For Beginners

Unlike the expanding window time series split, a sliding window
keeps the training size fixed and "slides" forward through time.

## How It Works

**How It Works:**
Notice: Training window is always the same size, just shifted forward.

**When to Use:**

- When old data becomes less relevant (concept drift)
- When you want fixed compute per training iteration
- When training on recent data only is preferred

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlidingWindowSplitter` | Initializes a new instance with default settings. |
| `SlidingWindowSplitter(Int32,Int32,Nullable<Int32>,Int32)` | Creates a new sliding window splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

