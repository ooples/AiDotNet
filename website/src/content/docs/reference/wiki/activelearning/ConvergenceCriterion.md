---
title: "ConvergenceCriterion<T>"
description: "Stopping criterion based on learning curve convergence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on learning curve convergence.

## For Beginners

This criterion analyzes the learning curve (how performance
changes with more data) and stops when adding more data shows diminishing returns.

## How It Works

**How It Works:**

**Key Insight:** Learning curves often follow a power law: performance = a - b*n^(-c)
where n is the number of samples. As n grows, improvement decreases.

**When to Use:**

**Reference:** Figueroa et al. "Predicting sample size required for classification performance" (BMC 2012)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConvergenceCriterion` | Initializes a new Convergence criterion with default parameters. |
| `ConvergenceCriterion(Int32,Double,Int32)` | Initializes a new Convergence criterion with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `Name` |  |
| `PredictedImprovement` | Gets the predicted improvement from additional samples. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

