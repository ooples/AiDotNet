---
title: "MediaPipeFacePreprocessor<T>"
description: "MediaPipe Face Mesh preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

MediaPipe Face Mesh preprocessor for ControlNet conditioning.

## For Beginners

This creates a wireframe-like mesh of facial features
(eyes, nose, mouth, jawline). ControlNet uses this to generate faces that
match the expression and structure of the original face.

## How It Works

Produces face mesh-like feature maps by detecting facial structure through
gradient analysis. The output highlights facial feature regions (eyes, nose,
mouth contours) that guide face-conditioned generation.

Reference: Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MediaPipeFacePreprocessor(Double)` | Initializes a new MediaPipe face mesh preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

