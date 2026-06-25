---
title: "GiniCoefficientMetric<T>"
description: "Computes Gini Coefficient (normalized): 2 * AUC - 1, measures discriminative ability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Gini Coefficient (normalized): 2 * AUC - 1, measures discriminative ability.

## For Beginners

Gini coefficient is directly related to AUC:

- Gini = 0: Random classifier (AUC = 0.5)
- Gini = 1: Perfect classifier (AUC = 1.0)
- Gini = -1: Perfectly wrong classifier (AUC = 0.0)

Common in credit scoring and insurance modeling. Sometimes called the "Gini index" (not to be confused
with Gini impurity used in decision trees).

## How It Works

Gini = 2 * AUC - 1

