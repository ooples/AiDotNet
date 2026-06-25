---
title: "AlphaCompositor<T>"
description: "Alpha compositing for layered diffusion outputs with transparency support."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Alpha compositing for layered diffusion outputs with transparency support.

## For Beginners

When generating an image in layers (like background + foreground +
effects), you need to combine them properly. AlphaCompositor handles the math of layering
images with transparency, just like how Photoshop stacks layers together. It ensures smooth
edges and correct color blending between layers.

## How It Works

Implements Porter-Duff alpha compositing operations for combining multiple generated
layers. Supports standard "over" compositing as well as additive and multiply blend modes.
Essential for multi-layer generation pipelines where foreground, background, and effects
layers are generated separately and composited into a final image.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AlphaCompositor(Boolean)` | Initializes a new alpha compositor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PremultipliedAlpha` | Gets whether inputs are expected in premultiplied alpha format. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompositeAdditive(Vector<>,Vector<>,Double,Double)` | Composites using additive blending. |
| `CompositeOver(Vector<>,Vector<>,Vector<>,Vector<>)` | Composites foreground over background using alpha blending (Porter-Duff "over" operation). |

