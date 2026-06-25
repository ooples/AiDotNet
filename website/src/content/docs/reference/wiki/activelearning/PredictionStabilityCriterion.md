---
title: "PredictionStabilityCriterion<T>"
description: "Stopping criterion based on prediction stability across iterations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on prediction stability across iterations.

## For Beginners

This criterion stops learning when the model's predictions
on the unlabeled pool stop changing significantly. Stable predictions indicate the
model has converged and won't benefit much from more labeled data.

## How It Works

**How It Works:**

**Stability Measures:**

**When to Use:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PredictionStabilityCriterion` | Initializes a new PredictionStability criterion with default parameters. |
| `PredictionStabilityCriterion(Double,Int32,StabilityMeasure)` | Initializes a new PredictionStability criterion with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentStability` | Gets the current stability measurement. |
| `Description` |  |
| `Name` |  |
| `StableIterations` | Gets the number of consecutive stable iterations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

