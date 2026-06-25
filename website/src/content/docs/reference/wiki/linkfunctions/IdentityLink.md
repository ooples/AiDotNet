---
title: "IdentityLink<T>"
description: "Identity link function: g(μ) = μ."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Identity link function: g(μ) = μ.

## For Beginners

Use this for standard linear regression where predictions
can be any real number. There's no transformation - what you predict is what you get.

## How It Works

The identity link is the canonical link for the Normal (Gaussian) distribution.
It makes no transformation, so the linear predictor equals the mean directly.

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

