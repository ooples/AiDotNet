---
title: "OnlineSlidingWindowSplitter<T>"
description: "Online sliding window splitter for streaming data with concept drift adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Online`

Online sliding window splitter for streaming data with concept drift adaptation.

## For Beginners

Unlike the landmark window, a sliding window maintains a fixed-size
training set that moves forward in time. Old samples are "forgotten" as new ones arrive.
This is essential when data patterns change over time (concept drift).

## How It Works

**How It Works:**

1. Maintain a window of the most recent N samples for training
2. Test on the next batch of samples
3. Slide the window forward, dropping oldest samples

**Window Strategies:**

- Fixed: Constant window size throughout
- Adaptive: Adjust window size based on performance
- Fading: Weight recent samples more heavily (conceptually)

**When to Use:**

- Non-stationary data (concept drift)
- Real-time systems with memory constraints
- When recent data is more relevant than old data

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineSlidingWindowSplitter(Int32,Int32,Int32,Int32,Int32)` | Creates a new online sliding window splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

