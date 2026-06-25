---
title: "FairnessEvaluatorBase<T>"
description: "Base class for all fairness evaluators that measure equitable treatment in models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Interpretability`

Base class for all fairness evaluators that measure equitable treatment in models.

## For Beginners

This is a foundation class that all fairness evaluators build upon.

Think of a fairness evaluator like a comprehensive audit:

- It examines your model's behavior across multiple dimensions of fairness
- It measures various fairness metrics (demographic parity, equal opportunity, etc.)
- It provides a complete picture of how equitably your model treats different groups

Different fairness evaluators might focus on different combinations of metrics, but they all
share common functionality. This base class provides that shared foundation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FairnessEvaluatorBase(Boolean)` | Initializes a new instance of the FairnessEvaluatorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsHigherFairnessBetter` | Gets a value indicating whether higher fairness scores represent better (more equitable) models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateFairness(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Int32,Vector<>)` | Evaluates fairness of a model by analyzing its predictions across different groups. |
| `GetFairnessMetrics(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Int32,Vector<>)` | Abstract method that must be implemented by derived classes to perform specific fairness evaluation logic. |
| `IsBetterFairnessScore(,)` | Determines whether a new fairness score represents better (more equitable) performance than the current best score. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_isHigherFairnessBetter` | Indicates whether higher fairness scores represent better (more equitable) models. |
| `_numOps` | Provides mathematical operations for the specific numeric type being used. |

