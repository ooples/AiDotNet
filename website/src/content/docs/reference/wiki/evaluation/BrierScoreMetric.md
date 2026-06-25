---
title: "BrierScoreMetric<T>"
description: "Computes Brier Score: mean squared error of probability predictions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Brier Score: mean squared error of probability predictions.

## For Beginners

Brier score measures how close predicted probabilities are to actual outcomes.

- Brier = 0: Perfect predictions (probabilities match outcomes exactly)
- Brier = 0.25: Random guessing for binary classification
- Brier = 1: Completely wrong (predicting 0% for all actual positives)

## How It Works

Brier Score = (1/N) * Σ(p_i - y_i)²

**Advantages:** Sensitive to calibration, proper scoring rule, penalizes overconfidence.

