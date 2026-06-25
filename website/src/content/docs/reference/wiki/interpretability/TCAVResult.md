---
title: "TCAVResult<T>"
description: "Represents the result of a TCAV analysis for a single concept."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a TCAV analysis for a single concept.

## For Beginners

This contains everything you need to know about
how a concept influences predictions for a specific class.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TCAVResult(String,Int32,Double[],Double,Double,Double,Boolean,Double,List<ConceptActivationVector<>>)` | Initializes a new TCAV result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CAVAccuracy` | Gets the mean CAV classifier accuracy. |
| `CAVs` | Gets all trained CAVs. |
| `ConceptName` | Gets the name of the concept being tested. |
| `IsSignificant` | Gets whether the result is statistically significant. |
| `MeanScore` | Gets the mean TCAV score across all runs. |
| `PValue` | Gets the p-value from statistical significance testing. |
| `StandardDeviation` | Gets the standard deviation of TCAV scores. |
| `TCAVScores` | Gets all TCAV scores from different CAV runs. |
| `TargetClass` | Gets the target class being analyzed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

