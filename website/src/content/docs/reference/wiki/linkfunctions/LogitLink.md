---
title: "LogitLink<T>"
description: "Logit link function: g(μ) = log(μ/(1-μ))."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Logit link function: g(μ) = log(μ/(1-μ)).

## For Beginners

Use this for binary classification (logistic regression).
It ensures your predictions are valid probabilities between 0 and 1.

The logit function is the log-odds:

- logit(0.5) = 0 (50-50 odds)
- logit(0.9) = 2.2 (9:1 odds)
- logit(0.1) = -2.2 (1:9 odds)

The inverse logit (sigmoid) converts back:

- sigmoid(0) = 0.5
- sigmoid(2.2) ≈ 0.9
- sigmoid(-2.2) ≈ 0.1

## How It Works

The logit link is the canonical link for the Binomial distribution.
It maps probabilities from (0,1) to (-∞,∞).

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

