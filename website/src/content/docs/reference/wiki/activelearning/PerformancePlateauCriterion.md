---
title: "PerformancePlateauCriterion<T>"
description: "Stopping criterion based on performance plateau detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on performance plateau detection.

## For Beginners

This criterion stops learning when the model's performance
(accuracy or validation score) stops improving significantly. Adding more labeled data
at this point is unlikely to help.

## How It Works

**How It Works:**

**Key Parameters:**

**Reference:** Early stopping is a common regularization technique in deep learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerformancePlateauCriterion` | Initializes a new PerformancePlateau criterion with default parameters. |
| `PerformancePlateauCriterion(Int32,Int32,Double,Boolean)` | Initializes a new PerformancePlateau criterion with specified parameters. |

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

