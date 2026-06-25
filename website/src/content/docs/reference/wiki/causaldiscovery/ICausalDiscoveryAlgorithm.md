---
title: "ICausalDiscoveryAlgorithm<T>"
description: "Interface for causal structure learning algorithms that discover Directed Acyclic Graphs (DAGs) from data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.CausalDiscovery`

Interface for causal structure learning algorithms that discover Directed Acyclic Graphs (DAGs) from data.

## For Beginners

This interface defines the contract for algorithms that figure out
cause-and-effect relationships from data. Given a dataset with multiple variables, the algorithm
produces a graph showing which variables directly cause changes in other variables.

For example, given data about weather, traffic, and commute time, a causal discovery algorithm
might find: Weather → Traffic → Commute Time (weather causes traffic, which causes longer commutes).

## How It Works

Causal discovery algorithms analyze observational data to infer the causal structure — a DAG where
edges represent direct causal relationships between variables. Unlike correlation analysis, these
algorithms attempt to determine the direction of causation.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the methodological category of this algorithm. |
| `Name` | Gets the display name of this algorithm. |
| `SupportsLatentConfounders` | Gets whether this algorithm can handle latent (unobserved) confounders. |
| `SupportsMixedData` | Gets whether this algorithm supports mixed (continuous and discrete) data types. |
| `SupportsNonlinear` | Gets whether this algorithm can discover nonlinear causal relationships. |
| `SupportsTimeSeries` | Gets whether this algorithm is designed for time series data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructure(Matrix<>,String[])` | Discovers causal structure from an observational data matrix. |
| `DiscoverStructure(Matrix<>,Vector<>,String[])` | Discovers causal structure with a designated target variable for directed analysis. |

