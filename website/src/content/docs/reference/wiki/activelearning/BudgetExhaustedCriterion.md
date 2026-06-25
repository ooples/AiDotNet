---
title: "BudgetExhaustedCriterion<T>"
description: "Stopping criterion based on labeling budget exhaustion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on labeling budget exhaustion.

## For Beginners

This is the most basic stopping criterion - it simply
checks if you've labeled as many samples as your budget allows.

## How It Works

**When to Use:**

**Implementation:** Compares TotalLabeled against MaxBudget in the context.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BudgetExhaustedCriterion` | Initializes a new BudgetExhausted criterion using context budget. |
| `BudgetExhaustedCriterion(Nullable<Int32>)` | Initializes a new BudgetExhausted criterion with a specific budget. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

