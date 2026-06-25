---
title: "ExplainedVarianceMetric<T>"
description: "Computes Explained Variance Score: proportion of variance in target explained by predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Explained Variance Score: proportion of variance in target explained by predictions.

## For Beginners

Similar to R² but doesn't penalize bias. EV = 1 means perfect variance
explanation, EV = 0 means no better than mean prediction. Can differ from R² if predictions are biased.

## How It Works

EV = 1 - Var(y - ŷ) / Var(y)

