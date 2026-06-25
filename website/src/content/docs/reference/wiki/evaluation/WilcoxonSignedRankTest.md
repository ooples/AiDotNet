---
title: "WilcoxonSignedRankTest<T>"
description: "Wilcoxon signed-rank test: non-parametric paired comparison test."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Wilcoxon signed-rank test: non-parametric paired comparison test.

## For Beginners

This is the non-parametric alternative to the paired t-test.
Use it when:

- Your differences are not normally distributed
- You have ordinal data
- Your sample size is small and you can't verify normality
- You want a more robust test (less sensitive to outliers)

## How It Works

**Common use in ML:** Comparing cross-validation scores of two models when
you're not sure the performance differences are normally distributed.

