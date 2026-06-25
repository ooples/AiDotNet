---
title: "R2ScoreMetric<T>"
description: "Computes R² (coefficient of determination): proportion of variance explained by the model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes R² (coefficient of determination): proportion of variance explained by the model.

## For Beginners

R² tells you what percentage of the variance in your target is
explained by the model.

- R² = 1: Perfect fit (model explains 100% of variance)
- R² = 0: Model is no better than predicting the mean
- R² < 0: Model is worse than predicting the mean

## How It Works

R² = 1 - (SS_res / SS_tot) = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

