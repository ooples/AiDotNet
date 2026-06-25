---
title: "RCDAlgorithm<T>"
description: "RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

RCD (Repetitive Causal Discovery) — LiNGAM extension for latent confounders.

## For Beginners

RCD is like DirectLiNGAM but more cautious — it checks whether
variables are truly "root causes" or might be influenced by hidden (unobserved) factors.
When it finds variables that can't be cleanly separated, it marks them as potentially
confounded rather than forcing a possibly wrong causal direction.

## How It Works

RCD extends DirectLiNGAM to handle latent (unobserved) confounders by iteratively
identifying "exogenous-like" variables whose residuals are mutually independent,
regressing them out, and repeating on the remaining variables.

**Algorithm:**

- Standardize the data
- Initialize the set of remaining variables U = {0, 1, ..., d-1}
- Repeat until U is empty:
- For each variable i in U, compute residuals after regressing out all discovered ancestors
- Compute pairwise independence (via mutual information) of residuals
- Identify the variable whose residuals are most independent of all others (exogenous)
- Add that variable to the causal ordering and record its causal coefficients
- Regress out the identified variable from all remaining variables
- If no sufficiently independent variable found, flag remaining as confounded and stop

Reference: Maeda and Shimizu (2020), "RCD: Repetitive Causal Discovery of
Linear Non-Gaussian Acyclic Models with Latent Confounders", AISTATS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RCDAlgorithm(CausalDiscoveryOptions)` | Initializes RCD with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeEntropyBasedMI(Double[0:,0:],Int32,Int32,Int32)` | Computes mutual information between two columns using entropy-based estimation. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `RegressCoefficient(Double[0:,0:],Int32,Int32,Int32)` | Computes the OLS regression coefficient of source predicting target. |

