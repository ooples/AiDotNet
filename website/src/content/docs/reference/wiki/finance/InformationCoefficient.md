---
title: "InformationCoefficient<T>"
description: "Information Coefficient (IC): the correlation between predicted and realized forward returns, plus its statistical significance (t-statistic, two-sided p-value) and the IC information ratio (ICIR) over a time series of per-period ICs."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Evaluation`

Information Coefficient (IC): the correlation between predicted and realized forward returns, plus
its statistical significance (t-statistic, two-sided p-value) and the IC information ratio (ICIR)
over a time series of per-period ICs.

## For Beginners

Imagine you predict tomorrow's stock moves for 500 stocks, then wait and
see what actually happened. The IC asks: "did the stocks I was most bullish on actually go up the
most?" An IC of +1 means a perfect match, 0 means no relationship (coin flip), and -1 means your
predictions were exactly backwards. Because even a tiny real edge is valuable, we also compute a
t-statistic and p-value to check that the IC is "really there" and not just luck. The ICIR is like a
Sharpe ratio for your forecasting skill: it rewards a positive IC that shows up *consistently*
period after period, not just on average.

## How It Works

The Information Coefficient is the north-star metric of return-forecasting research: it measures how
well a model's predictions rank/track the returns that actually occurred. Both the Pearson (linear)
and Spearman (rank) flavors are provided. An IC of about 0.03-0.05 that is statistically significant
already represents a real, exploitable edge in liquid markets.

## Methods

| Method | Summary |
|:-----|:--------|
| `InformationRatio(IReadOnlyList<>)` | Summarizes a time series of per-period ICs into mean IC, IC standard deviation, and the IC information ratio: ICIR = meanIC / stdIC · sqrt(periods). |
| `LogGamma(Double)` | Lanczos approximation of ln(Gamma(z)) for z > 0. |
| `Pearson(IReadOnlyList<>,IReadOnlyList<>)` | Pearson (linear) correlation between predicted and realized returns. |
| `Ranks(Double[])` | Tie-averaged ("fractional") ranks of a sample, 1-based. |
| `RegularizedIncompleteBeta(Double,Double,Double)` | Regularized incomplete beta I_x(a, b) via the Lentz continued fraction (NR-style). |
| `Significance(,Int32)` | Two-sided significance test for an IC value over `n` observations. |
| `Spearman(IReadOnlyList<>,IReadOnlyList<>)` | Spearman rank correlation: Pearson correlation of the (tie-averaged) ranks of the two series. |
| `StudentTwoSidedPValue(Double,Int32)` | Two-sided p-value for a Student-t statistic with `df` degrees of freedom, via the regularized incomplete beta function: p = I_{df/(df+t^2)}(df/2, 1/2). |

