---
title: "MarkovBlanketAlgorithm<T>"
description: "Markov Blanket (Grow-Shrink) Algorithm — discovers the Markov blanket of each variable."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

Markov Blanket (Grow-Shrink) Algorithm — discovers the Markov blanket of each variable.

## For Beginners

Think of a variable's Markov blanket as a "shield" — if you know
all variables in the blanket, no other variable can provide additional information about
the target. This algorithm finds that shield by adding helpful variables (growing) and
then removing redundant ones (shrinking).

## How It Works

The Grow-Shrink algorithm identifies the Markov blanket of each variable through two phases:

- **Growing:** Add variables that increase conditional mutual information with the target.
- **Shrinking:** Remove variables that become conditionally independent given the rest.

Reference: Margaritis and Thrun (1999), "Bayesian Network Induction via Local Neighborhoods".

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarkovBlanketAlgorithm(CausalDiscoveryOptions,Int32,Double,Int32)` | Creates a new Markov Blanket algorithm instance. |

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

