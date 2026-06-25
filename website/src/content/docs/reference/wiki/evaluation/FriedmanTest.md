---
title: "FriedmanTest<T>"
description: "Friedman test: non-parametric test for comparing multiple classifiers across multiple datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Friedman test: non-parametric test for comparing multiple classifiers across multiple datasets.

## For Beginners

The Friedman test is the standard statistical test for comparing
multiple machine learning algorithms across multiple datasets. It's recommended by Demsar (2006)
for ML algorithm comparison.

- Ranks each algorithm within each dataset
- Tests if the average ranks are significantly different
- Non-parametric: doesn't assume normal distribution of scores

## How It Works

**Typical workflow:**

- Run k algorithms on n datasets (e.g., via cross-validation)
- Use Friedman test to see if there's any significant difference
- If significant, use post-hoc tests (e.g., Nemenyi) to find which pairs differ

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([][],Double)` | Performs the Friedman test. |

