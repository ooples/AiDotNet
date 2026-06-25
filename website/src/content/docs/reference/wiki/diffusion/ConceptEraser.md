---
title: "ConceptEraser<T>"
description: "Concept erasure for removing unwanted concepts from diffusion model representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Safety`

Concept erasure for removing unwanted concepts from diffusion model representations.

## For Beginners

Imagine the model's understanding of concepts as directions
in a high-dimensional space. Concept erasure finds the "direction" corresponding to
an unwanted concept and removes it, like erasing one axis from a coordinate system.
The model can still generate everything else, but it loses the ability to produce
the erased concept.

## How It Works

Concept erasure removes specific concepts (e.g., artistic styles, identities, or
unsafe content) from a diffusion model's internal representations by projecting
embeddings onto the null space of the target concept direction. This prevents the
model from generating content related to the erased concept.

Reference: Gandikota et al., "Erasing Concepts from Diffusion Models", ICCV 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConceptEraser(Double,Boolean)` | Initializes a new concept eraser. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConceptCount` | Gets the number of registered concept directions. |
| `ErasureStrength` | Gets the erasure strength. |
| `PreserveGeneralCapability` | Gets whether general capability preservation is enabled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddConceptDirection(Vector<>)` | Registers a concept direction to be erased. |
| `ComputeConceptPresence(Vector<>,Int32)` | Computes the concept presence score for an embedding (0 = absent, 1 = fully present). |
| `EraseFromEmbedding(Vector<>)` | Erases registered concepts from an embedding by projecting onto the null space. |

