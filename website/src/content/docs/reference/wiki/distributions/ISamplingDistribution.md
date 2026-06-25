---
title: "ISamplingDistribution<T>"
description: "Extends parametric distributions with sampling capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Distributions`

Extends parametric distributions with sampling capabilities.

## For Beginners

Sampling means generating random numbers that follow
a specific pattern (distribution). For example, if you sample from a Normal
distribution with mean 0 and variance 1, most samples will be close to 0,
and samples far from 0 (like 3 or -3) will be rare.

## How It Works

Sampling distributions support generating random variates from the distribution.
This is essential for Monte Carlo methods, simulation, and Bayesian inference.

## Methods

| Method | Summary |
|:-----|:--------|
| `Sample` | Generates a single random sample using a default random number generator. |
| `Sample(Int32)` | Generates multiple random samples using a default random number generator. |
| `Sample(Random)` | Generates a single random sample from the distribution. |
| `Sample(Random,Int32)` | Generates multiple random samples from the distribution. |

