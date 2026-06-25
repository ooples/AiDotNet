---
title: "ReciprocalLink<T>"
description: "Inverse (reciprocal) link function: g(μ) = 1/μ."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Inverse (reciprocal) link function: g(μ) = 1/μ.

## For Beginners

Use this when doubling a predictor halves the response
(inverse relationship). Common in:

- Time-to-event data with constant hazard
- Some physics/engineering applications

Note: The inverse link can produce negative predictions, so ensure
the linear predictor stays positive for meaningful results.

## How It Works

The inverse link is the canonical link for the Gamma distribution.
It's useful when the relationship is inversely proportional.

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

