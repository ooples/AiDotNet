---
title: "BootstrapTest<T>"
description: "Bootstrap-based hypothesis test for comparing two models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Bootstrap-based hypothesis test for comparing two models.

## For Beginners

Bootstrap testing uses resampling to test significance:

- Repeatedly samples with replacement from your data
- Computes the statistic of interest on each sample
- Builds a distribution to test your hypothesis
- No assumptions about the underlying distribution

## How It Works

**Advantages:**

- Works for any metric, not just specific test statistics
- No distributional assumptions
- Can compute confidence intervals for any statistic

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BootstrapTest(Int32,Nullable<Int32>)` | Initializes the bootstrap test. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCI([],[],Double)` | Computes bootstrap confidence interval for the mean difference. |
| `Test([],[])` | Tests if two sets of scores differ significantly using bootstrap. |

