---
title: "LogLink<T>"
description: "Log link function: g(μ) = log(μ)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Log link function: g(μ) = log(μ).

## For Beginners

Use this when your response variable is always positive:

- Count data (number of events)
- Positive continuous values (income, time, distance)

The log link ensures predictions are always positive (after inverse transform).
A one-unit increase in a predictor multiplies the expected response by exp(β).

## How It Works

The log link is the canonical link for the Poisson distribution and is also
commonly used with Gamma and other positive-valued distributions.
It maps positive values to (-∞,∞).

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

