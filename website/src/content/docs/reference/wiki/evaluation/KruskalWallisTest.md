---
title: "KruskalWallisTest<T>"
description: "Kruskal-Wallis H test for comparing multiple independent groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Kruskal-Wallis H test for comparing multiple independent groups.

## For Beginners

Kruskal-Wallis is the non-parametric alternative to one-way ANOVA:

- Compares medians of 3+ independent groups
- No assumption of normal distribution
- Uses ranks instead of raw values
- Tests if at least one group differs from others

## How It Works

**Difference from Friedman test:**

- Kruskal-Wallis: independent groups (different samples)
- Friedman: paired/matched groups (same samples, different treatments)

## Methods

| Method | Summary |
|:-----|:--------|
| `DunnPostHoc([][])` | Performs Dunn's post-hoc test for pairwise comparisons. |
| `Test([][])` | Tests if multiple groups have significantly different distributions. |

