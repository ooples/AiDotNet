---
title: "FBetaScoreMetric<T>"
description: "Computes F-beta score: weighted harmonic mean of precision and recall."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes F-beta score: weighted harmonic mean of precision and recall.

## For Beginners

F-beta lets you weight precision vs recall:

- beta = 1: F1 score (equal weight)
- beta = 2: F2 score (recall twice as important as precision)
- beta = 0.5: F0.5 score (precision twice as important as recall)

## How It Works

F_beta = (1 + beta²) * (precision * recall) / (beta² * precision + recall)

