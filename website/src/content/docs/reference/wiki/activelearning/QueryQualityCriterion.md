---
title: "QueryQualityCriterion<T>"
description: "Stopping criterion based on quality of remaining query candidates."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.StoppingCriteria`

Stopping criterion based on quality of remaining query candidates.

## For Beginners

This criterion stops learning when the informativeness
scores of the best remaining samples fall below a threshold. If no informative
samples remain, there's little benefit to continuing.

## How It Works

**How It Works:**

**Intuition:** Early in active learning, the query strategy finds highly
informative samples. As learning progresses, the most informative samples get labeled,
and remaining samples become less valuable.

**Key Parameters:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryQualityCriterion` | Initializes a new QueryQuality criterion with default parameters. |
| `QueryQualityCriterion(Double,Boolean)` | Initializes a new QueryQuality criterion with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentMaxScore` | Gets the current maximum query score. |
| `Description` |  |
| `InitialMaxScore` | Gets the initial maximum query score (for relative mode). |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetProgress(ActiveLearningContext<>)` |  |
| `Reset` |  |
| `ShouldStop(ActiveLearningContext<>)` |  |

