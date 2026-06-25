---
title: "SGRACEEraser<T>"
description: "S-GRACE: Style-aware GRACE for erasing artistic styles from diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Safety`

S-GRACE: Style-aware GRACE for erasing artistic styles from diffusion models.

## For Beginners

Removing an artistic style from a model is harder than removing
an object because styles affect how everything looks, not just what appears. S-GRACE
is designed specifically for this challenge — it finds and removes the "style knowledge"
distributed throughout the model's layers, so the model can't mimic the erased style
but still generates high-quality images in other styles.

## How It Works

S-GRACE extends the GRACE (GRadient-based Concept Erasure) method to handle artistic
style concepts, which are more distributed across model weights than object concepts.
It uses style-specific gradient directions and multi-layer editing to effectively
remove artistic style associations while preserving content generation quality.

Reference: Pham et al., "Robust Concept Erasure Using Task Vectors", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SGRACEEraser(Double,Int32,Double)` | Initializes a new S-GRACE eraser. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ErasureRate` | Gets the erasure rate. |
| `NumIterations` | Gets the number of erasure iterations. |
| `PreservationThreshold` | Gets the preservation threshold. |
| `StyleCount` | Gets the number of registered style vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddStyleVector(Vector<>)` | Registers a style task vector for erasure. |
| `ComputeErasureUpdate(Vector<>,Vector<>)` | Computes the erasure update for model parameters using the task vector negation approach. |
| `MeasureStylePresence(Vector<>,Vector<>,Int32)` | Measures style presence as cosine similarity between parameter delta and style vector. |

