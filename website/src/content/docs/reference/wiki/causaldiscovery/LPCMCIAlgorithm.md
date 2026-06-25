---
title: "LPCMCIAlgorithm<T>"
description: "LPCMCI — Latent PCMCI for time series with hidden confounders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

LPCMCI — Latent PCMCI for time series with hidden confounders.

## For Beginners

LPCMCI is the most advanced version of PCMCI. It works even when
there are hidden variables affecting the ones you can measure. The trade-off is that
some edges may be uncertain in direction (shown with circle marks).

## How It Works

LPCMCI extends PCMCI to handle latent confounders by combining ideas from FCI
(ancestral graph representation) with PCMCI's condition selection and MCI testing.
It outputs a time series PAG (partial ancestral graph) instead of a DAG.

**Algorithm:**

- Run PCMCI condition selection to find preliminary parents for each variable
- Apply FCI-style skeleton thinning: iteratively test edges conditioning on subsets

of the selected parents, removing edges that become independent

- Apply orientation rules that account for possible latent confounders:

collider orientation, discriminating paths, and temporal constraints

- Mark edges with ambiguous orientation (possible latent confounder) as bidirected
- Compute summary graph: collapse lags, keep max absolute weight per pair

Reference: Gerhardus and Runge (2020), "High-recall causal discovery for autocorrelated
time series with latent confounders", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsLatentConfounders` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

