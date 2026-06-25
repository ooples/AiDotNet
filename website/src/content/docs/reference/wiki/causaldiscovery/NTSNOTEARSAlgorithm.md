---
title: "NTSNOTEARSAlgorithm<T>"
description: "NTS-NOTEARS — Nonstationary Time Series NOTEARS."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.TimeSeries`

NTS-NOTEARS — Nonstationary Time Series NOTEARS.

## For Beginners

Regular time series methods assume the causal relationships stay
the same forever. NTS-NOTEARS can detect when relationships change — for example,
a market regime shift where the causes of stock prices change.

## How It Works

NTS-NOTEARS extends DYNOTEARS to handle nonstationary time series where the causal
structure may change over time. It partitions the data into segments using a variance-based
change-point detector, learns a separate NOTEARS-style DAG for each segment, and produces
a summary graph by taking the maximum absolute edge weight across segments, preserving
regime-specific causal effects.

**Algorithm:**

- Partition time series into K segments using variance-based change-point detection
- For each segment, learn a contemporaneous DAG via NOTEARS with DYNOTEARS-style

lagged terms and the augmented Lagrangian acyclicity constraint

- Aggregate segment-level DAGs into a summary graph: edge weight = max |w_k| over

segments where that edge appears, preserving regime-specific causal effects

Reference: Sun et al. (2021), "NTS-NOTEARS: Learning Nonparametric DBN Structure
from Nonstationary Time Series".

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

