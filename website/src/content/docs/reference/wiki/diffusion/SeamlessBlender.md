---
title: "SeamlessBlender<T>"
description: "Seamless blending for panoramic and tiled diffusion generation with overlap regions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Seamless blending for panoramic and tiled diffusion generation with overlap regions.

## For Beginners

When creating large images by stitching together smaller patches
(like a panorama), the edges of each patch need to blend smoothly together. SeamlessBlender
handles this transition, making sure there are no visible lines or color jumps where
patches meet — similar to how panorama photo apps stitch multiple photos together.

## How It Works

Handles the smooth blending of overlapping patches in panoramic, tiled, or outpainting
generation pipelines. Supports linear ramp, cosine, and Gaussian blending profiles for
the overlap regions. Ensures seamless transitions without visible seams at patch boundaries.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SeamlessBlender(BlendProfile,Int32)` | Initializes a new seamless blender. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OverlapSize` | Gets the overlap size in pixels/elements. |
| `Profile` | Gets the blend profile type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BlendOverlap(Vector<>,Vector<>)` | Blends two overlapping patches in the overlap region. |
| `GenerateBlendWeights` | Generates a 1D blend weight ramp for the configured overlap size. |

