---
title: "RandomEffect<T>"
description: "Represents a random effect specification in a mixed-effects model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression.MixedEffects`

Represents a random effect specification in a mixed-effects model.

## For Beginners

Random effects are like "adjustments" for each group in your data.

For example, if studying student test scores across different schools:

- Fixed effect: Overall relationship between study time and scores
- Random effect: Each school might have its own baseline score level

Random effects model this group-level variation properly, accounting for the fact that
students in the same school are more similar to each other than to students from other schools.

## How It Works

Random effects model group-level variation around population-level (fixed) effects.
They capture correlation within groups and account for unobserved heterogeneity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomEffect(String,Int32)` | Initializes a new instance of RandomEffect for a random intercept. |
| `RandomEffect(String,Int32,Int32[],Boolean)` | Initializes a new instance of RandomEffect with random slopes. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CovarianceMatrix` | Gets or sets the variance-covariance matrix of random effects. |
| `Dimension` | Gets the dimension of the random effect (1 for intercept only, more with slopes). |
| `GroupCoefficients` | Gets or sets the estimated random effect coefficients for each group. |
| `GroupColumnIndex` | Gets or sets the grouping variable column index. |
| `IsRandomIntercept` | Gets or sets whether this is a random intercept. |
| `Name` | Gets or sets the name of this random effect. |
| `NumberOfGroups` | Gets the number of unique groups in this random effect. |
| `RandomSlopeColumns` | Gets or sets the feature column indices for random slopes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetGroupEffect(Double)` | Gets the random effect vector for a specific group. |
| `SetGroupEffect(Double,Vector<>)` | Sets the random effect vector for a specific group. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |

