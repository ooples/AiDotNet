---
title: "IPageSegmenter<T>"
description: "Interface for page segmentation models that identify different regions in document pages."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for page segmentation models that identify different regions in document pages.

## For Beginners

When you look at a document page, you can easily identify
different sections - titles, paragraphs, images, tables. Page segmentation teaches
computers to do the same thing, labeling each region with its type.

Example usage:

## How It Works

Page segmentation models divide a document page into semantic regions like
text blocks, figures, tables, headers, footers, and captions.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedRegionTypes` | Gets the region types this segmenter can detect. |
| `SupportsInstanceSegmentation` | Gets whether this segmenter performs instance segmentation (separate instances of same type). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSegmentationMask(Tensor<>)` | Gets the pixel-level segmentation mask. |
| `SegmentPage(Tensor<>)` | Segments a document page into semantic regions. |
| `SegmentPage(Tensor<>,Double)` | Segments a document page with a custom confidence threshold. |

