---
title: "InfoTheoreticBase<T>"
description: "Base class for information-theoretic causal discovery algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.InformationTheoretic`

Base class for information-theoretic causal discovery algorithms.

## For Beginners

These methods measure how much "information" flows between variables.
If knowing variable X tells you a lot about variable Y (beyond what other variables tell
you), that suggests X has a causal influence on Y.

## How It Works

Information-theoretic methods use entropy, mutual information, and transfer entropy to
discover causal relationships. These measures quantify the amount of information one
variable provides about another, either unconditionally or conditioned on other variables.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `KNeighbors` | Number of nearest neighbors for MI estimation (Kraskov method). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInfoOptions(CausalDiscoveryOptions)` | Applies information-theoretic options. |
| `ComputeEntropy(Matrix<>,Int32)` | Computes Shannon entropy (Gaussian approximation) of a column. |
| `ComputeGaussianMI(Matrix<>,Int32,Int32)` | Computes Gaussian mutual information between two columns. |

