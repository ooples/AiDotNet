---
title: "TSFCIAlgorithm<T>"
description: "tsFCI — time series Fast Causal Inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

tsFCI — time series Fast Causal Inference.

## For Beginners

tsFCI is like FCI but for time series. It can discover causal
relationships even when there are hidden variables affecting the observed ones,
using the fact that "the future cannot cause the past" to help figure out direction.

## How It Works

tsFCI adapts the FCI algorithm for time series data, allowing for the discovery of
causal relationships in the presence of latent (unmeasured) confounders. It uses
temporal ordering constraints (the future cannot cause the past) combined with
conditional independence testing on lagged variables to orient edges and identify
latent confounders.

**Algorithm:**

- Build a time-expanded graph with nodes X_t^(i) for each variable i at each lag
- Start with complete graph, remove edges via conditional independence tests on lagged data
- Apply temporal constraints: remove edges where cause lag > effect lag
- Orient remaining edges using FCI rules adapted for temporal ordering
- Mark edges with potential latent confounders as bidirected
- Compute summary graph: collapse lagged edges to contemporaneous with max weight

Reference: Entner and Hoyer (2010), "On Causal Discovery from Time Series Data
using FCI", PGM.

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

