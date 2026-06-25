---
title: "VARLiNGAMAlgorithm<T>"
description: "VAR-LiNGAM — Vector Autoregressive LiNGAM for time series causal discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

VAR-LiNGAM — Vector Autoregressive LiNGAM for time series causal discovery.

## For Beginners

This algorithm finds causal relationships in time series data
that work at different time scales. It can detect both "X causes Y right now"
(contemporaneous) and "yesterday's X causes today's Y" (lagged) relationships.

## How It Works

VAR-LiNGAM combines VAR (Vector Autoregression) with LiNGAM to discover both
contemporaneous (same time-step) and lagged (across time-steps) causal relationships.

**Model:** X(t) = B₀ X(t) + B₁ X(t-1) + ... + Bₖ X(t-k) + e(t)
where B₀ encodes contemporaneous effects and B₁...Bₖ encode lagged effects.

Reference: Hyvarinen et al. (2010), "Estimation of a Structural Vector Autoregression
Model Using Non-Gaussianity", JMLR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

