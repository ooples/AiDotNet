---
title: "CLAHE<T>"
description: "Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

## For Beginners

Regular histogram equalization can make noise very visible.
CLAHE works on small regions and limits how much contrast it adds, so you get better
detail without amplifying noise. It's widely used in medical imaging.

## How It Works

CLAHE improves upon standard histogram equalization by dividing the image into tiles
and equalizing each tile independently, with a clip limit to prevent noise amplification.
The tile boundaries are blended using bilinear interpolation for smooth results.
Based on Zuiderveld (1994).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLAHE(Double,Int32,Double)` | Creates a new CLAHE augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClipLimit` | Gets the clip limit for contrast limiting. |
| `TileGridSize` | Gets the number of tiles in each dimension. |

