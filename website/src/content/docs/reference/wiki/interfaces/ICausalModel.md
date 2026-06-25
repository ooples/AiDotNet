---
title: "ICausalModel<T>"
description: "Interface for causal inference models (meta-learners)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for causal inference models (meta-learners).

## For Beginners

Causal inference models estimate the causal effect of a treatment
on an outcome. Unlike prediction, we want to know "what would happen if we applied treatment X?"
This is called the treatment effect.

## How It Works

**Key concepts:**

- **Treatment:** The intervention we're studying (e.g., a drug, a marketing campaign)
- **Outcome:** The result we measure (e.g., health, sales)
- **CATE:** Conditional Average Treatment Effect - effect for specific subgroups
- **ATE:** Average Treatment Effect - overall average effect

**Meta-learners:**

- **S-Learner:** Single model with treatment as feature
- **T-Learner:** Two separate models for treatment/control
- **X-Learner:** Cross-fitting approach for heterogeneous effects

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateAverageTreatmentEffect(Matrix<>)` | Estimates the Average Treatment Effect (ATE) across the population. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates the Conditional Average Treatment Effect (CATE) for subjects. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model to observational data. |
| `PredictControl(Matrix<>)` | Predicts the outcome under control. |
| `PredictTreated(Matrix<>)` | Predicts the outcome under treatment. |

