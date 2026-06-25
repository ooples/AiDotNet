---
title: "NelsonAalenEstimator<T>"
description: "Implements the Nelson-Aalen estimator for non-parametric cumulative hazard function estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements the Nelson-Aalen estimator for non-parametric cumulative hazard function estimation.

## For Beginners

The Nelson-Aalen estimator calculates the cumulative hazard function H(t),
which represents the accumulated risk of an event up to time t. Unlike Kaplan-Meier which estimates
survival probability S(t), Nelson-Aalen estimates H(t) directly. They're related by S(t) = exp(-H(t)).

## How It Works

**How it works:**

- At each event time t, add d(t)/n(t) to the cumulative hazard
- d(t) = number of events at time t
- n(t) = number of subjects at risk just before time t

**Variance estimation:** The Nelson-Aalen estimator uses the variance formula:
Var(H(t)) = sum over event times of d(t)/n(t)^2

**When to use:**

- When you want to estimate cumulative hazard directly
- As input to other models that work with cumulative hazard
- When comparing hazard rates across groups

**Reference:** Nelson (1972), Aalen (1978)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NelsonAalenEstimator` | Creates a new Nelson-Aalen estimator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CumulativeHazard` | Gets the cumulative hazard function values at event times. |
| `Variance` | Gets the variance estimates at event times. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Nelson-Aalen estimator to survival data. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function. |
| `GetParameters` |  |
| `InterpolateCumulativeHazard(Double)` | Interpolates cumulative hazard at a specific time. |
| `Predict(Matrix<>)` | Predicts survival probability at median event time. |
| `PredictCumulativeHazard(Vector<>,Matrix<>)` | Predicts cumulative hazard at specified times (override to use direct estimate). |
| `PredictHazardRatio(Matrix<>)` | Returns hazard ratios (all 1s for non-parametric estimator). |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified times. |
| `Serialize` |  |
| `SetParameters(Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cumulativeHazard` | The cumulative hazard values at each event time. |
| `_variance` | The variance estimates at each event time. |

