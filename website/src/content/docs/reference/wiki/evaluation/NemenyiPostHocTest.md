---
title: "NemenyiPostHocTest<T>"
description: "Nemenyi post-hoc test: pairwise comparisons after Friedman test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Nemenyi post-hoc test: pairwise comparisons after Friedman test.

## For Beginners

After the Friedman test shows a significant difference exists
among multiple algorithms, the Nemenyi test helps identify which specific pairs differ.

- Controls family-wise error rate (FWER)
- Computes critical difference (CD) for significance
- Two algorithms are significantly different if |rank_i - rank_j| > CD

## How It Works

**Critical Difference Diagram:** The results can be visualized in a CD diagram,
a standard visualization in ML literature for algorithm comparison.

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([][],Double)` | Performs Nemenyi post-hoc test on multiple algorithms. |

