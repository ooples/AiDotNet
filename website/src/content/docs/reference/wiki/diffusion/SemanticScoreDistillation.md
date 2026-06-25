---
title: "SemanticScoreDistillation<T>"
description: "Semantic Score Distillation (SemanticSDS) for semantically-guided 3D generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Semantic Score Distillation (SemanticSDS) for semantically-guided 3D generation.

## For Beginners

Sometimes SDS produces 3D objects that look nice but don't match
the text description well (semantic drift). SemanticSDS splits the optimization into
"does it mean the right thing?" and "does it look good?", optimizing both separately.
This ensures the 3D model of a "red ceramic mug" actually looks like a mug, not just
a red blob.

## How It Works

SemanticSDS enhances score distillation by incorporating semantic understanding from
vision-language models (e.g., CLIP). It decomposes the SDS gradient into semantic and
appearance components, allowing independent control over what the 3D object represents
versus how it looks. This reduces semantic drift and produces 3D objects that better
match the text prompt's meaning.

Reference: Adapted from semantic-aware score distillation techniques combining CLIP
guidance with diffusion model scores for text-to-3D generation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SemanticScoreDistillation(IDiffusionModel<>,Double,Double,Double)` | Initializes a new SemanticSDS instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AppearanceWeight` | Gets the weight for the appearance component. |
| `GuidanceScale` | Gets the guidance scale. |
| `SemanticWeight` | Gets the weight for the semantic component. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Vector<>)` | Computes the semantic SDS gradient with decomposed components. |
| `ComputeSemanticAlignment(Vector<>,Vector<>)` | Computes the semantic alignment score between render embedding and text embedding. |

