---
title: "CopulaSynthOptions<T>"
description: "Configuration options for Copula-Based Synthesis, a statistical method that models the joint distribution of features by fitting marginal distributions individually and coupling them with a copula function."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Copula-Based Synthesis, a statistical method that models
the joint distribution of features by fitting marginal distributions individually
and coupling them with a copula function.

## For Beginners

Copula synthesis is like building a recipe in two steps:

Step 1 — Learn each feature's shape separately:
"Age is normally distributed around 40"
"Income follows a log-normal distribution"

Step 2 — Learn how features relate to each other:
"When Age is high, Income tends to be high too"
"Education and Income are strongly correlated"

This separation makes the method very flexible: you can model each feature
with whatever distribution fits best, and the copula captures how they move together.

Example:

## How It Works

Copula synthesis separates two concerns:

- **Marginals**: Each feature's individual distribution (fitted independently)
- **Copula**: The dependency structure between features (fitted via rank correlations)

## Properties

| Property | Summary |
|:-----|:--------|
| `BandwidthMultiplier` | Gets or sets the KDE bandwidth multiplier. |
| `CopulaType` | Gets or sets the copula family to use for dependency modeling. |
| `NumKDEPoints` | Gets or sets the number of points for kernel density estimation of marginals. |

