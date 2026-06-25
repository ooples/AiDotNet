---
title: "CAMUVAlgorithm<T>"
description: "CAM-UV — Causal Additive Model with Unobserved Variables."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

CAM-UV — Causal Additive Model with Unobserved Variables.

## For Beginners

Sometimes two variables appear related not because one causes the
other, but because a hidden third variable causes both. CAM-UV detects these situations
by checking: if neither direction X→Y nor Y→X fits cleanly, there might be a hidden
common cause. It marks such pairs as "confounded" rather than forcing a causal direction.

## How It Works

CAM-UV extends CAM to handle latent (unobserved) confounders. It discovers the causal
structure among observed variables even when some common causes are hidden. The algorithm:

- Fits pairwise additive noise models between all variable pairs.
- Identifies potential latent confounders by detecting pairs where residuals in

both directions show high dependence (neither direction fits well).

- Marks bidirectional edges for pairs with suspected latent confounders.
- Orients remaining edges using the standard ANM asymmetry criterion.

Reference: Maeda and Shimizu (2021), "Causal Additive Models with Unobserved Variables".

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

