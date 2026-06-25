---
title: "LandmarkWindowSplitter<T>"
description: "Landmark window splitter for online learning with growing training set."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.Online`

Landmark window splitter for online learning with growing training set.

## For Beginners

A landmark window keeps all historical data from a fixed starting
point (the "landmark"). As new data arrives, the training set grows but always starts
from the same point. This is useful when concept drift is minimal and all history is relevant.

## How It Works

**How It Works:**

1. Define a landmark (start point), typically timestamp 0
2. Training set includes all data from landmark to current time
3. Test on the next batch of samples
4. Training window expands over time

**Comparison:**

- Sliding window: Fixed size, moves forward (forgets old data)
- Landmark window: Variable size, keeps all history

**When to Use:**

- When all historical data remains relevant
- Stationary data distributions
- Long-term pattern learning

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LandmarkWindowSplitter(Int32,Int32,Int32,Int32,Int32)` | Creates a new landmark window splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

