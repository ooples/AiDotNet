---
title: "KaplanMeierEstimator<T>"
description: "Implements the Kaplan-Meier estimator for non-parametric survival analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SurvivalAnalysis`

Implements the Kaplan-Meier estimator for non-parametric survival analysis.

## For Beginners

Kaplan-Meier is the simplest and most widely used survival method.
It creates a "staircase" survival curve that shows the probability of survival over time.

How it works:

1. Sort all observation times
2. At each time point where an event occurs:
- Count how many subjects are "at risk" (still in the study)
- Count how many had the event
- Survival probability = (at risk - events) / at risk
3. Cumulative survival = product of all these probabilities up to time t

Example:

- Time 0: 100 patients, all alive → S(0) = 1.0
- Time 1: 100 at risk, 10 die → S(1) = 1.0 × (90/100) = 0.90
- Time 2: 90 at risk (some left the study), 5 die → S(2) = 0.90 × (85/90) = 0.85

Key features:

- No assumptions about the shape of survival (non-parametric)
- Handles censoring naturally
- Does NOT use patient features - same curve for everyone

For comparing groups or using features, use Cox Proportional Hazards instead.

References:

- Kaplan & Meier (1958). "Nonparametric Estimation from Incomplete Observations"

## How It Works

The Kaplan-Meier estimator is a non-parametric method for estimating the survival function
from lifetime data. It doesn't use covariates - it estimates a single survival curve for
all subjects in the dataset.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KaplanMeierEstimator` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of the same type. |
| `FitSurvivalCore(Matrix<>,Vector<>,Vector<Int32>)` | Fits the Kaplan-Meier estimator to the survival data. |
| `GetBaselineSurvival(Vector<>)` | Gets the baseline survival function at specified time points. |
| `GetEventTimes` | Gets the event times used in the survival curve. |
| `GetNumberAtRisk` | Gets the number at risk at each event time. |
| `GetNumberEvents` | Gets the number of events at each event time. |
| `GetParameters` | Gets all model parameters as a single vector. |
| `GetSurvivalProbabilities` | Gets the survival probability at each event time. |
| `Predict(Matrix<>)` | Standard prediction - returns survival probability at a reference time. |
| `PredictHazardRatio(Matrix<>)` | Predicts hazard ratios (all 1.0 for Kaplan-Meier since it doesn't use covariates). |
| `PredictSurvivalProbability(Matrix<>,Vector<>)` | Predicts survival probabilities at specified time points. |
| `SetParameters(Vector<>)` | Sets the parameters for this model. |
| `WithParameters(Vector<>)` | Creates a new instance of the model with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numberAtRisk` | Stores the number at risk at each event time. |
| `_numberEvents` | Stores the number of events at each event time. |
| `_survivalProbabilities` | Stores the survival probability at each event time. |

