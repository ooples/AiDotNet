---
title: "IStoppingCriterion<T>"
description: "Interface for stopping criteria in active learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for stopping criteria in active learning.

## For Beginners

Active learning loops can be stopped based on various criteria
beyond just exhausting the labeling budget. Early stopping can save resources when
additional labels won't significantly improve the model.

## How It Works

**Common Stopping Criteria:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of when this criterion triggers. |
| `Name` | Gets the name of the stopping criterion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` | Gets a progress indicator (0 to 1) showing how close to stopping. |
| `Reset` | Resets the criterion to its initial state. |
| `ShouldStop(ActiveLearningContext<>)` | Checks whether the stopping criterion is met. |

