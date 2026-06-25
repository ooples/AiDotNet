---
title: "CausalDiscoverySelector<T>"
description: "Feature selector that uses any causal discovery algorithm to select features based on causal relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery`

Feature selector that uses any causal discovery algorithm to select features based on causal relationships.

## For Beginners

Instead of selecting features by correlation or mutual information, this
selector uses causal discovery to find features that actually CAUSE the target variable (or are
caused by it). This often leads to better, more robust models because causal features remain
predictive even when the data distribution changes.

## How It Works

This wrapper allows any `ICausalDiscoveryAlgorithm` to be used as a feature selector
within the preprocessing pipeline. It discovers the causal graph, then selects the top features
that have the strongest causal connections to the target variable.

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalDiscoverySelector(CausalDiscoveryAlgorithmType,Int32,Double,CausalDiscoveryOptions,Int32[])` | Creates a new CausalDiscoverySelector using the specified algorithm type. |
| `CausalDiscoverySelector(ICausalDiscoveryAlgorithm<>,Int32,Double,Int32[])` | Creates a new CausalDiscoverySelector using a pre-created algorithm instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CausalGraph` | Gets the discovered causal graph from the most recent Fit() call. |
| `ConnectionStrengths` | Gets the connection strengths of each feature to the target. |
| `SelectedIndices` | Gets the selected feature indices. |
| `SupportsInverseTransform` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Fits the selector by discovering the causal graph and selecting features connected to the target. |
| `FitCore(Matrix<>)` |  |
| `FitTransform(Matrix<>,Vector<>)` | Fits the selector and transforms the data in one step. |
| `GetFeatureNamesOut(String[])` |  |
| `GetSupportMask` | Gets a boolean mask indicating which input features are selected. |
| `TransformCore(Matrix<>)` |  |

