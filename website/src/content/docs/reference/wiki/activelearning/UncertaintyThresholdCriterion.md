---
title: "UncertaintyThresholdCriterion<T>"
description: "Stopping criterion based on model uncertainty reaching a threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on model uncertainty reaching a threshold.

## For Beginners

This criterion stops learning when the model becomes
sufficiently confident on the unlabeled data. If the model has low uncertainty
on remaining samples, labeling them is unlikely to help much.

## How It Works

**How It Works:**

**Key Parameters:**

**Common Uncertainty Measures:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UncertaintyThresholdCriterion` | Initializes a new UncertaintyThreshold criterion with default parameters. |
| `UncertaintyThresholdCriterion(Double,Double)` | Initializes a new UncertaintyThreshold criterion with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentAverageUncertainty` |  |
| `Description` |  |
| `FractionConfident` |  |
| `Name` |  |
| `UncertaintyThreshold` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

