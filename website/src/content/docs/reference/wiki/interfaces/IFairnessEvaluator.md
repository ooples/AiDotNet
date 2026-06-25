---
title: "IFairnessEvaluator<T>"
description: "Defines an interface for evaluating fairness in machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for evaluating fairness in machine learning models.

## How It Works

**For Beginners:** This interface helps measure how fair a machine learning model is.

Fairness in machine learning means treating all groups of people equally. A fair model
makes similar decisions for people with similar qualifications, regardless of sensitive
attributes like race, gender, or age.

This interface provides methods to:

- Evaluate how fair a model is across different groups
- Measure various aspects of fairness (demographic parity, equal opportunity, etc.)
- Compare different models to find which one is most equitable

The fairness score measures how equitable the model is. Important points:

- Higher fairness scores can indicate more equitable treatment (depends on the metric)
- Multiple fairness metrics exist because fairness can be defined in different ways
- We can use these measurements to improve our models and ensure equal treatment

## Properties

| Property | Summary |
|:-----|:--------|
| `IsHigherFairnessBetter` | Indicates whether higher fairness scores represent better (more equitable) models. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateFairness(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Int32,Vector<>)` | Evaluates fairness of a model by analyzing its predictions across different groups. |
| `IsBetterFairnessScore(,)` | Compares two fairness scores and determines if the current score represents better (more equitable) performance. |

