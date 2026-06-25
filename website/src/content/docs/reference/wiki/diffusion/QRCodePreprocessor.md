---
title: "QRCodePreprocessor<T>"
description: "QR code pattern preprocessor for ControlNet conditioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Preprocessing`

QR code pattern preprocessor for ControlNet conditioning.

## For Beginners

This preprocessor cleans up a QR code image so that
ControlNet can embed the QR code pattern into generated artwork. The result
is a high-contrast black-and-white image that clearly shows the QR pattern.

## How It Works

Enhances QR code or grid-like binary patterns in images by applying adaptive
thresholding and contrast enhancement. The output preserves high-contrast
black/white patterns suitable for QR code-conditioned generation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QRCodePreprocessor(Int32)` | Initializes a new QR code preprocessor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputChannels` |  |
| `OutputControlType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Transform(Tensor<>)` |  |

