---
title: "PCMCIPlusAlgorithm<T>"
description: "PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

PCMCI+ — extension of PCMCI that also discovers contemporaneous causal links.

## For Beginners

PCMCI only finds "yesterday's X causes today's Y" relationships.
PCMCI+ also finds "today's X causes today's Y" relationships, which are important
when variables influence each other faster than the measurement interval.

## How It Works

PCMCI+ extends PCMCI to handle both lagged AND contemporaneous (same time-step)
causal links by adding a skeleton discovery and orientation step for lag-0 effects.
It applies the same Fisher z-test based conditional independence testing at lag 0
conditioned on the lagged parents discovered by PCMCI.

Reference: Runge (2020), "Discovering Contemporaneous and Lagged Causal Relations
in Autocorrelated Nonlinear Time Series Datasets", UAI.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeContemporaneousPartialCorrelation(Matrix<>,Int32,Int32,List<Int32>,Int32,Int32)` | Computes partial correlation between variables i(t) and j(t) at lag 0, conditioned on lagged parents via OLS residualization. |
| `DiscoverStructureCore(Matrix<>)` |  |

