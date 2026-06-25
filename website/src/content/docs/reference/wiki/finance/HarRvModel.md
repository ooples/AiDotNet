---
title: "HarRvModel<T>"
description: "HAR-RV — the Heterogeneous AutoRegressive model of Realized Volatility (Corsi, 2009, \"A Simple Approximate Long-Memory Model of Realized Volatility\", Journal of Financial Econometrics 7(2))."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Volatility`

HAR-RV — the Heterogeneous AutoRegressive model of Realized Volatility (Corsi, 2009,
"A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics 7(2)).

## For Beginners

Volatility clusters and has "long memory" — calm and turbulent stretches
persist. HAR-RV predicts tomorrow's variance from how volatile the last day, last week, and last month
were, fit by plain least-squares. Simple, robust, and hard to beat.

## How It Works

HAR-RV forecasts next-period realized variance as a linear function of realized variance averaged over
three horizons — daily, weekly (5), and monthly (22) — capturing volatility's long-memory with a simple
regression:

Corsi estimates this by **ordinary least squares**, so this model extends `RegressionBase`
(AiDotNet's classical OLS base) rather than the neural model base — faithful to the paper's estimator.
It is the canonical baseline for the one signal the platform found rigorously predictable (volatility),
and produces the forecast realized vol consumed by the vol-edge options strategy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HarRvModel(RegressionOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Creates a HAR-RV model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildHarDesign(IReadOnlyList<>)` | Builds the HAR design from a realized-variance series: each row is [RV_t, mean(RV over last 5), mean(RV over last 22)] and the target is RV_{t+1}. |
| `CreateNewInstance` |  |
| `FitRealizedVariance(IReadOnlyList<>)` | Fits directly on a realized-variance series (builds the HAR design, then OLS). |
| `ForecastAnnualizedVol(IReadOnlyList<>,Double)` | Forecasts next-period ANNUALIZED realized vol = √(variance × periodsPerYear). |
| `ForecastNextVariance(IReadOnlyList<>)` | Forecasts next-period realized VARIANCE from the latest history. |
| `ForecastVolFromReturns(IReadOnlyList<Double>,Double)` | Convenience: fit + forecast annualized vol straight from a per-period RETURN series, using squared returns as the realized-variance proxy (RV_t = r_t²). |
| `HarRow(IReadOnlyList<>,Int32)` | The HAR feature row at time `t`: [daily RV, weekly avg, monthly avg]. |
| `Train(Matrix<>,Vector<>)` | Fits the HAR coefficients by OLS (normal equations) — the estimator in Corsi (2009). |

## Fields

| Field | Summary |
|:-----|:--------|
| `MonthlyWindow` | Monthly aggregation horizon (trading days) from Corsi (2009). |
| `WeeklyWindow` | Weekly aggregation horizon (trading days) from Corsi (2009). |

