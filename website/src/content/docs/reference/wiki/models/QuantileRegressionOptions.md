---
title: "QuantileRegressionOptions<T>"
description: "Configuration options for Quantile Regression, a technique that enables prediction of specific quantiles of the conditional distribution rather than just the conditional mean."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Quantile Regression, a technique that enables prediction of specific
quantiles of the conditional distribution rather than just the conditional mean.

## For Beginners

Quantile Regression helps predict specific percentiles of possible outcomes, not just the average outcome.

Think about salary predictions:

- Regular regression might tell you "the average salary for this job is $75,000"
- But Quantile Regression could tell you:
- "10% of people in this job earn less than $50,000" (10th percentile)
- "Half of people in this job earn less than $70,000" (median or 50th percentile)
- "90% of people in this job earn less than $120,000" (90th percentile)

What this technique does:

- It focuses on specific slices of the data distribution
- Instead of minimizing squared errors (as in mean regression)
- It minimizes a different loss function that depends on which quantile you want
- This gives you insight into different parts of the outcome distribution

This is especially useful when:

- The outcomes aren't evenly distributed around the average
- You're interested in extreme cases (very high or low values)
- Different factors might affect different parts of the distribution differently
- You want to understand risk or uncertainty better

For example, in healthcare, knowing that a treatment reduces the risk of severe complications
(the high quantile) is different information than knowing it reduces the average symptom severity.

This class lets you configure how the quantile regression algorithm operates.

## How It Works

Quantile Regression extends traditional regression methods by estimating conditional quantiles
of the response variable. While standard regression estimates the conditional mean E(Y|X),
Quantile Regression can estimate any conditional quantile Q(a|X) for a ? (0,1), including
medians (a = 0.5) and other percentiles. This technique provides a more comprehensive view of the
relationship between variables, allowing for the analysis of the full conditional distribution.
It is particularly valuable when the conditional distribution is non-Gaussian, skewed, or when
outliers are present. Quantile Regression is also robust to heteroscedasticity (non-constant variance)
and can reveal how different parts of the distribution respond differently to predictor variables.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate for the optimization algorithm. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the optimization algorithm. |
| `Quantile` | Gets or sets the quantile to be estimated by the regression model. |

