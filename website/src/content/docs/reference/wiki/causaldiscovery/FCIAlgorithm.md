---
title: "FCIAlgorithm<T>"
description: "FCI (Fast Causal Inference) — constraint-based discovery with latent confounders."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ConstraintBased`

FCI (Fast Causal Inference) — constraint-based discovery with latent confounders.

## For Beginners

Sometimes two variables appear related not because one causes the other,
but because a hidden third variable causes both. FCI can detect this pattern.

## How It Works

FCI extends the PC algorithm to handle latent (unmeasured) variables. It outputs a
Partial Ancestral Graph (PAG) which uses different edge marks (→, ←→, o→) to indicate
whether edges are definitely directed, bidirected (due to a latent common cause), or uncertain.

Reference: Spirtes, Glymour, and Scheines (2000), "Causation, Prediction, and Search".

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

