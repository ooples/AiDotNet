---
title: "ISurvivalModel<T>"
description: "Interface for survival analysis models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for survival analysis models.

## For Beginners

Survival analysis models time-to-event data, where we're interested
in predicting when an event will occur (e.g., customer churn, equipment failure, patient survival).
A key challenge is "censoring" - when we don't observe the event for some subjects.

## How It Works

**Key concepts:**

- **Survival function S(t):** Probability of surviving beyond time t
- **Hazard function h(t):** Instantaneous risk of event at time t
- **Cumulative hazard H(t):** Accumulated risk up to time t
- **Censoring:** When event is not observed (e.g., study ends before event)

**Common models:**

- **Kaplan-Meier:** Non-parametric survival curve estimation
- **Nelson-Aalen:** Non-parametric cumulative hazard estimation
- **Cox PH:** Semi-parametric proportional hazards model
- **AFT:** Accelerated failure time models (Weibull, Log-Normal)

## Properties

| Property | Summary |
|:-----|:--------|
| `BaselineSurvival` | Gets the baseline survival function values at event times. |
| `EventTimes` | Gets the unique event times from the training data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Vector<>,Vector<>,Matrix<>)` | Fits the survival model to time-to-event data. |
| `PredictCumulativeHazard(Vector<>,Matrix<>)` | Predicts cumulative hazard at specified times. |
| `PredictMedianSurvivalTime(Matrix<>)` | Gets the estimated median survival time. |
| `PredictRisk(Matrix<>)` | Predicts risk scores for subjects (higher = higher risk). |
| `PredictSurvival(Vector<>,Matrix<>)` | Predicts survival probability at specified times. |

