---
title: "Deskew<T>"
description: "Deskew - Document deskewing utility using Hough transform-based angle detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.Document`

Deskew - Document deskewing utility using Hough transform-based angle detection.

## For Beginners

When documents are scanned, they often end up slightly rotated.
This tool detects and corrects that rotation:

- Detects dominant line angles in the document
- Uses Hough transform for robust angle detection
- Applies rotation correction
- Preserves document content

Key features:

- Automatic skew angle detection
- Configurable angle range
- High accuracy for text documents
- Works with various document types

Example usage:

## How It Works

Deskew detects and corrects rotation in scanned documents using Hough transform
analysis to find dominant line angles, then applies inverse rotation to straighten the document.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Deskew` | Creates a new Deskew instance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectSkewAngle(Tensor<>,Double)` | Detects the skew angle of a document image. |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the deskew utility. |
| `Process(Tensor<>,Double)` | Processes an image to correct skew. |
| `RotateImage(Tensor<>,Double)` | Rotates an image by the specified angle. |

