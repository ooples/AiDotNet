---
title: "ConfidenceIntervalMethod"
description: "Specifies the method for computing confidence intervals on evaluation metrics."
section: "API Reference"
---

`Enums` · `AiDotNet.Evaluation.Enums`

Specifies the method for computing confidence intervals on evaluation metrics.

## For Beginners

When you calculate a metric like accuracy = 85%, that's just a point
estimate. The "true" accuracy might be anywhere from 82% to 88%. A confidence interval
tells you this range. Different methods compute this range differently:

- **Bootstrap methods**: Resample your data many times to see how the metric varies
- **Analytical methods**: Use mathematical formulas (faster but more assumptions)

Bootstrap methods are generally preferred as they make fewer assumptions about your data.

## How It Works

Confidence intervals quantify the uncertainty in metric estimates. Different methods
make different assumptions about the underlying distribution and have varying accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `BCaBootstrap` | Bias-Corrected and Accelerated (BCa) bootstrap: Corrects for bias and skewness. |
| `BasicBootstrap` | Basic bootstrap (reverse percentile): Reflects the bootstrap distribution. |
| `BayesianCredible` | Bayesian credible interval: Uses posterior distribution from Bayesian inference. |
| `ClopperPearsonExact` | Clopper-Pearson exact interval: Conservative exact interval for proportions. |
| `DeLong` | DeLong method: Specialized method for AUROC confidence intervals. |
| `NormalApproximation` | Normal approximation: Simple z-interval using normal distribution. |
| `PercentileBootstrap` | Percentile bootstrap: Uses percentiles of the bootstrap distribution directly. |
| `StudentizedBootstrap` | Studentized (bootstrap-t) bootstrap: Uses t-statistics for pivoting. |
| `TDistribution` | t-distribution interval: Uses t-distribution instead of normal. |

