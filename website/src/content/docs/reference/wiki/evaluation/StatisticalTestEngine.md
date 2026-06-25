---
title: "StatisticalTestEngine<T>"
description: "Engine for performing statistical tests on model comparison results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Engines`

Engine for performing statistical tests on model comparison results.

## For Beginners

When comparing machine learning models, you need statistical tests
to determine if one model is truly better or if the difference is just due to random chance.
This engine provides methods for:

- Comparing two models (paired t-test, Wilcoxon, McNemar)
- Comparing multiple models (Friedman test)
- Post-hoc analysis (Nemenyi test)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StatisticalTestEngine` | Initializes the statistical test engine with all available tests. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareModels([][],String[],Double)` | Performs full comparison of multiple models with automatic test selection. |
| `CompareMultipleModelsFriedman([][],Double)` | Compares multiple models using the Friedman test. |
| `CompareTwoClassifiersMcNemar([],[],Double)` | Compares two classifiers using McNemar's test on their predictions. |
| `CompareTwoModelsPairedTTest([],[],Double)` | Compares two models using paired t-test (parametric). |
| `CompareTwoModelsWilcoxon([],[],Double)` | Compares two models using Wilcoxon signed-rank test (non-parametric). |
| `PostHocNemenyi([][],Double)` | Performs Nemenyi post-hoc test after significant Friedman test. |

