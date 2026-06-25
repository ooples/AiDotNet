---
title: "CompositeCriterion<T>"
description: "Composite stopping criterion that combines multiple criteria."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Composite stopping criterion that combines multiple criteria.

## For Beginners

Sometimes you want to stop based on multiple conditions.
For example, stop if you run out of budget OR if performance plateaus. This composite
criterion lets you combine multiple criteria with AND/OR logic.

## How It Works

**Combination Modes:**

**Common Combinations:**

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CompositeCriterion` | Initializes a new empty CompositeCriterion with Any mode. |
| `CompositeCriterion(CombinationMode)` | Initializes a new empty CompositeCriterion with specified mode. |
| `CompositeCriterion(IEnumerable<IStoppingCriterion<>>,CombinationMode)` | Initializes a CompositeCriterion with specified criteria and mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Criteria` |  |
| `Description` |  |
| `Mode` | Gets the combination mode for this composite. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCriterion(IStoppingCriterion<>)` |  |
| `All(IStoppingCriterion<>[])` | Creates a composite criterion that stops when ALL of the given criteria are met. |
| `Any(IStoppingCriterion<>[])` | Creates a composite criterion that stops when ANY of the given criteria is met. |
| `GetProgress(ActiveLearningContext<>)` |  |
| `RemoveCriterion(IStoppingCriterion<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

