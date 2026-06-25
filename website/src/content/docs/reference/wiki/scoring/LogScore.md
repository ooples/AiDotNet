---
title: "LogScore<T>"
description: "Logarithmic scoring rule (negative log likelihood)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Scoring`

Logarithmic scoring rule (negative log likelihood).

## For Beginners

The log score asks "how surprised was the model by what
actually happened?" If the model assigned high probability to the true outcome,
it gets a low (good) score. If it assigned low probability to what happened,
it gets a high (bad) penalty.

For example, if you predict 90% chance of rain and it rains, that's a good prediction.
But if you predict 1% chance of rain and it rains, that's a terrible prediction and
you get heavily penalized.

## How It Works

The logarithmic score (also called log loss or cross-entropy) is the negative
log probability density/mass assigned to the observed value. It's the most
widely used proper scoring rule due to its connection to maximum likelihood.

Score = -log(p(y)) where p(y) is the density at observation y.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsMinimized` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Score(IParametricDistribution<>,)` |  |
| `ScoreGradient(IParametricDistribution<>,)` |  |

