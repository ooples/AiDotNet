---
title: "UnlabeledPoolExhaustedCriterion<T>"
description: "Stopping criterion that triggers when the unlabeled pool is exhausted."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion that triggers when the unlabeled pool is exhausted.

## For Beginners

This is a simple but important criterion - it stops
active learning when there are no more unlabeled samples to query. This is the
natural endpoint when all data has been labeled.

## How It Works

**When to Use:**

**Optional Threshold:** Can also stop when pool falls below a minimum
size, useful when very small pools aren't worth querying.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UnlabeledPoolExhaustedCriterion` | Initializes a new UnlabeledPoolExhausted criterion that stops at zero remaining. |
| `UnlabeledPoolExhaustedCriterion(Int32)` | Initializes a new UnlabeledPoolExhausted criterion with minimum threshold. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `MinimumRemaining` | Gets the minimum pool size before stopping. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

