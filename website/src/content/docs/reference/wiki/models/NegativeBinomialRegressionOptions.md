---
title: "NegativeBinomialRegressionOptions<T>"
description: "Configuration options for Negative Binomial Regression, a statistical model used for count data that exhibits overdispersion (variance exceeding the mean)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Negative Binomial Regression, a statistical model used for count data
that exhibits overdispersion (variance exceeding the mean).

## For Beginners

Negative Binomial Regression is a specialized technique for analyzing
count data - data where you're counting how many times something happens.

While Poisson regression is commonly used for count data, it assumes that the mean and variance are equal.
In real-world data, we often see more variability than Poisson allows for:

Think of it like predicting daily customer counts at a restaurant:

- Some days might have 50 customers, others 150, with an average of 100
- Poisson would expect most days to be fairly close to 100
- But real data often shows more extreme values (very busy days, very slow days)
- Negative Binomial can handle this extra variability

This model is particularly useful when:

- You're counting events (visits, purchases, accidents, etc.)
- Your data shows "clumping" or extra variation
- Some counts are much higher or lower than the average would suggest

This class lets you configure how the model learns from your data to make accurate predictions
despite this extra variability.

## How It Works

Negative Binomial Regression extends Poisson regression by introducing an additional parameter
that allows the variance to exceed the mean, making it suitable for overdispersed count data.
This model is appropriate when analyzing count outcomes (like number of events, occurrences, or items)
that show greater variability than would be expected under a Poisson distribution. The model is
typically fitted using maximum likelihood estimation, optimized through iterative methods such as
Fisher scoring or Newton-Raphson iterations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type to use when solving the weighted least squares problem. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the optimization algorithm should stop. |

