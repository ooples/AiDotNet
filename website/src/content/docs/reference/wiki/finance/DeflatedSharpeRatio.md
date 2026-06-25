---
title: "DeflatedSharpeRatio<T>"
description: "López de Prado's Deflated Sharpe Ratio (DSR): the probability that a strategy's *true* Sharpe ratio is positive, after deflating the observed Sharpe for (a) the number of strategy configurations tried (multiple-testing / selection bias) and…"
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Evaluation`

López de Prado's Deflated Sharpe Ratio (DSR): the probability that a strategy's *true* Sharpe
ratio is positive, after deflating the observed Sharpe for (a) the number of strategy configurations
tried (multiple-testing / selection bias) and (b) non-normality (skew and excess kurtosis) of returns.

## For Beginners

If you try 1,000 random trading rules, one of them will look great purely
by luck. The DSR corrects for that: it asks "given that I tested N strategies, how surprising is this
Sharpe ratio, really?" and answers with a probability that the strategy's edge is real (true Sharpe
> 0). It also accounts for the fact that real returns have fat tails and asymmetry, which make a
raw Sharpe ratio less trustworthy. Higher DSR = more confidence the result is not a fluke.

## How It Works

When many strategies are backtested, the best observed Sharpe is upward-biased simply by chance. The
DSR compares the observed Sharpe against the *expected maximum* Sharpe that `N` independent
trials of a truly skill-less process would produce, and expresses the result as a probability in
[0, 1] using the Sharpe-ratio estimator's standard error (which itself depends on skew and kurtosis).
A DSR above ~0.95 is the usual bar for declaring a discovery genuine.

## Methods

| Method | Summary |
|:-----|:--------|
| `Compute(Double,Int32,Int32,Double,Double)` | Computes the Deflated Sharpe Ratio. |
| `Erf(Double)` | Abramowitz & Stegun 7.1.26 erf approximation (\|error\| < 1.5e-7). |
| `ExpectedMaxSharpe(Int32)` | Expected maximum Sharpe ratio (per-observation, i.e. |
| `NormalCdf(Double)` | Standard normal CDF via the erf relation, with a high-accuracy erf approximation. |
| `NormalQuantile(Double)` | Inverse standard-normal CDF (probit) via Acklam's rational approximation. |

