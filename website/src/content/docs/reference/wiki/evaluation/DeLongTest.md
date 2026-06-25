---
title: "DeLongTest<T>"
description: "DeLong's test for comparing two ROC curves."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

DeLong's test for comparing two ROC curves.

## For Beginners

DeLong's test compares the AUC of two classifiers:

- Tests if the difference in AUC is statistically significant
- Specifically designed for comparing ROC curves
- Non-parametric and doesn't assume normal distribution
- Can handle correlated data (same test set for both models)

## How It Works

**When to use:**

- Comparing two binary classifiers on the same dataset
- When you want to know if AUC improvement is significant
- Medical diagnostic test comparisons

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([],[],[])` | Performs DeLong's test comparing two sets of predicted probabilities. |

