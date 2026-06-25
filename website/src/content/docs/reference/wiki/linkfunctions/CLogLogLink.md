---
title: "CLogLogLink<T>"
description: "Complementary log-log link function: g(μ) = log(-log(1-μ))."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LinkFunctions`

Complementary log-log link function: g(μ) = log(-log(1-μ)).

## For Beginners

Use this when:

- You're modeling the probability of an event that becomes increasingly likely
- The probability approaches 1 faster than it approaches 0
- Survival analysis with complementary log-log model

Unlike logit (symmetric), cloglog is asymmetric:

- cloglog(0.5) ≈ -0.37 (not 0 like logit)
- Approaches 0 slowly from below
- Approaches 1 quickly from above

This is the canonical link for the extreme value (Gumbel) distribution.

## How It Works

The complementary log-log link is asymmetric around 0.5, unlike logit and probit.
It's useful for modeling probabilities with asymmetric behavior.

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

