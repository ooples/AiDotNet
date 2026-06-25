---
title: "TimeBudgetCriterion<T>"
description: "Stopping criterion based on time budget exhaustion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on time budget exhaustion.

## For Beginners

This criterion stops learning when the total time
spent on active learning exceeds a specified budget. Useful when you have
time constraints on annotation or training.

## How It Works

**When to Use:**

**Considerations:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeBudgetCriterion` | Initializes a new TimeBudget criterion with a 1-hour default. |
| `TimeBudgetCriterion(TimeSpan)` | Initializes a new TimeBudget criterion with specified time limit. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `MaxTime` | Gets the maximum allowed time. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

