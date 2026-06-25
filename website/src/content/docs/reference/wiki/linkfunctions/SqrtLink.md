---
title: "SqrtLink<T>"
description: "Square root link function: g(μ) = √μ."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Square root link function: g(μ) = √μ.

## For Beginners

Use this when:

- You have count data but want to moderate extreme predictions
- The log link produces predictions that are too extreme

The square root link is gentler than the log link:

- log(100) = 4.6, so exp(5) = 148
- sqrt(100) = 10, so 10² = 100

This means changes in the linear predictor have a more moderate
effect on predictions.

## How It Works

The square root link is often used with Poisson count data as an alternative
to the log link. It provides variance stabilization.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsCanonical` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InverseLink()` |  |
| `InverseLinkDerivative()` |  |
| `Link()` |  |
| `LinkDerivative()` |  |
| `Variance()` |  |

