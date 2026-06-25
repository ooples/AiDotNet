---
title: "LeveneTest<T>"
description: "Levene's test for equality of variances across groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Levene's test for equality of variances across groups.

## For Beginners

Levene's test checks if groups have equal variance:

- Important assumption for many statistical tests (t-test, ANOVA)
- More robust than Bartlett's test to non-normality
- Uses deviations from group means or medians

## How It Works

**Variants:**

- Mean-based: Original Levene's test
- Median-based: Brown-Forsythe test (more robust)
- Trimmed mean: Compromise between the two

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LeveneTest(LeveneTest<>.CenterType)` | Initializes Levene's test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([][])` | Tests if multiple groups have equal variances. |

