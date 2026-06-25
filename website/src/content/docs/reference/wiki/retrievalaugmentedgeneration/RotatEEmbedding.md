---
title: "RotatEEmbedding<T>"
description: "RotatE embedding model: models relations as rotations in complex vector space."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

RotatE embedding model: models relations as rotations in complex vector space.

## For Beginners

Instead of translation (TransE), RotatE rotates entity vectors.
Each relation is a rotation angle per dimension. This handles more relation patterns:

- Symmetric: "married_to" (A→B and B→A) uses 180° rotation
- Antisymmetric: "parent_of" (if A→B, then NOT B→A)
- Composition: "grandparent_of" = "parent_of" + "parent_of" (rotations compose)

## How It Works

RotatE (Sun et al., 2019) represents entities as complex vectors and relations as
element-wise rotations: t = h ∘ r, where |r_i| = 1 (unit modulus). This captures
symmetric, antisymmetric, inversion, and composition patterns.

Complex numbers are represented as paired real/imaginary T[] arrays to keep the generic T
type parameter (System.Numerics.Complex is double-only).

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDistanceBased` |  |

