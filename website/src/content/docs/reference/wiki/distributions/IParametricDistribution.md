---
title: "IParametricDistribution<T>"
description: "Defines a parametric probability distribution with learnable parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Distributions`

Defines a parametric probability distribution with learnable parameters.

## For Beginners

A parametric distribution is like a template for
describing uncertainty. For example, a Normal distribution is defined by
just two numbers: mean (center) and variance (spread). Once you know these
parameters, you know everything about the distribution.

## How It Works

Parametric distributions are fully specified by a fixed set of parameters
(e.g., mean and variance for Normal, shape and rate for Gamma).
This interface provides methods for computing probability densities,
cumulative distributions, and gradients needed for gradient-based learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Mean` | Gets the mean (expected value) of the distribution. |
| `NumParameters` | Gets the number of parameters that define this distribution. |
| `ParameterNames` | Gets the parameter names for this distribution. |
| `Parameters` | Gets or sets the distribution parameters as a vector. |
| `StdDev` | Gets the standard deviation of the distribution. |
| `Variance` | Gets the variance of the distribution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Cdf()` | Computes the cumulative distribution function (CDF) at point x. |
| `Clone` | Creates a copy of this distribution with the same parameters. |
| `FisherInformation` | Computes the Fisher Information Matrix for the distribution. |
| `GradLogPdf()` | Computes the gradient of the log PDF with respect to each parameter. |
| `InverseCdf()` | Computes the inverse CDF (quantile function) for probability p. |
| `LogPdf()` | Computes the log probability density function at point x. |
| `Pdf()` | Computes the probability density function (PDF) at point x. |

