---
title: "BootstrapConfidenceInterval<T>"
description: "Bootstrap confidence intervals for a statistic of a return series."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Evaluation`

Bootstrap confidence intervals for a statistic of a return series. Supports the IID bootstrap and the
stationary (Politis-Romano) block bootstrap, which preserves serial dependence in time-series returns.
Built-in statistics include the mean and the (per-observation) Sharpe ratio; any custom statistic can
be supplied.

## For Beginners

Suppose your strategy's Sharpe ratio is 1.2. Is that "1.2 give or take 0.1",
or "1.2 give or take 0.8"? The bootstrap answers this by repeatedly re-sampling your return history
(with replacement) thousands of times, re-computing the statistic each time, and looking at the spread.
The middle 95% of those values is your confidence interval. We resample whole *blocks* of returns
(not isolated days) so that streaks and momentum in the data are respected. A seeded random generator
is passed in so the result is exactly reproducible.

## How It Works

A point estimate (e.g. a backtest's Sharpe ratio) is just one number; the bootstrap quantifies how much
it would wobble under resampling, yielding a percentile confidence interval without distributional
assumptions. The stationary bootstrap resamples random-length blocks (geometric length distribution)
rather than single observations, so autocorrelation in returns is not destroyed.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(IReadOnlyList<>,Random,Func<IReadOnlyList<Double>,Double>,Double,Int32,Double)` | Computes a bootstrap percentile confidence interval. |
| `MeanStatistic(IReadOnlyList<Double>)` | Mean of a return sample (per-observation). |
| `Percentile(Double[],Double)` | Linear-interpolated percentile of a pre-sorted array (percentile in [0, 100]). |
| `SharpeStatistic(IReadOnlyList<Double>)` | Per-observation Sharpe ratio (mean / sample-stdev) of a return sample. |

