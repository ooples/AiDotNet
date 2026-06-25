---
title: "MixedEffectsModelOptions"
description: "Configuration options for Mixed-Effects (Hierarchical/Multilevel) Models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Mixed-Effects (Hierarchical/Multilevel) Models.

## For Beginners

Mixed-effects models are perfect when your data has a natural grouping structure:

**Examples:**

- Students within schools: Students in the same school share characteristics
- Patients within hospitals: Patients at the same hospital have similar care patterns
- Measurements over time: Repeated measurements from the same person are correlated
- Products within brands: Products from the same brand share brand-level qualities

**Why not just use regular regression?**
Regular regression assumes all observations are independent. But students at the same
school are more similar to each other than students at different schools. Ignoring this
leads to overconfident predictions and incorrect conclusions.

**Two types of effects:**

- **Fixed effects:** Population-level patterns (e.g., "on average, studying 1 more hour

increases test scores by 5 points")

- **Random effects:** Group-level variations (e.g., "some schools have higher baseline

scores than others")

The model estimates both the population patterns AND how much groups vary from this pattern.

## How It Works

Mixed-effects models contain both fixed effects (population-level parameters) and random
effects (group-level variations). They're essential for analyzing hierarchical or clustered
data where observations are not independent.

## Properties

| Property | Summary |
|:-----|:--------|
| `CenterFeatures` | Gets or sets whether to center features before fitting. |
| `CovarianceStructure` | Gets or sets the covariance structure for random effects. |
| `IncludeRandomIntercept` | Gets or sets whether to include random intercepts. |
| `IncludeRandomSlopes` | Gets or sets whether to include random slopes. |
| `MaxIterations` | Gets or sets the maximum number of iterations for optimization. |
| `MinObservationsPerGroup` | Gets or sets the minimum number of observations per group. |
| `OptimizationMethod` | Gets or sets the optimization method. |
| `RandomSlopeFeatures` | Gets or sets the indices of features to include as random slopes (if enabled). |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `UseRobustStandardErrors` | Gets or sets whether to use robust standard errors. |

