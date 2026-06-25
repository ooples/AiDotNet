---
title: "CochranQTest<T>"
description: "Cochran's Q test for comparing multiple classifiers on the same dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Cochran's Q test for comparing multiple classifiers on the same dataset.

## For Beginners

Cochran's Q test is an extension of McNemar's test:

- Compares 3+ classifiers on the same binary classification task
- Tests if all classifiers have the same error rate
- Non-parametric test for matched samples
- Uses chi-square distribution for p-value

## How It Works

**When to use:**

- Comparing multiple classifiers on the same test set
- Binary classification only (use Friedman for continuous scores)
- When samples are matched (same data across all classifiers)

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([][],[])` | Tests if multiple classifiers have significantly different error rates. |

