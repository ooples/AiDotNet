---
title: "KGEmbeddingType"
description: "Specifies the type of knowledge graph embedding model to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of knowledge graph embedding model to use.

## For Beginners

Each embedding type has strengths for different relationship patterns:

- TransE: Good for one-to-one relations (born_in, capital_of)
- RotatE: Handles symmetric, antisymmetric, inversion, and composition patterns
- ComplEx: Best for symmetric and antisymmetric relations
- DistMult: Best for symmetric relations (similar_to, married_to)
- TemporalTransE: TransE with time-awareness for facts that change over time

## How It Works

Knowledge graph embeddings map entities and relations to continuous vector spaces,
enabling mathematical reasoning about graph structure. Different models capture
different types of relational patterns.

## Fields

| Field | Summary |
|:-----|:--------|
| `ComplEx` | ComplEx: Complex-valued embedding using Hermitian dot product. |
| `DistMult` | DistMult: Bilinear diagonal model scoring Σ(h_k · r_k · t_k). |
| `RotatE` | RotatE: Rotation-based embedding in complex space where t = h ∘ r. |
| `TemporalTransE` | TemporalTransE: Time-aware TransE with discretized time bins. |
| `TransE` | TransE: Translational embedding where h + r ≈ t. |

