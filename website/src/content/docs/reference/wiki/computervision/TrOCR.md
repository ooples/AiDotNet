---
title: "TrOCR<T>"
description: "TrOCR (Transformer-based OCR) for text recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.OCR.Recognition`

TrOCR (Transformer-based OCR) for text recognition.

## For Beginners

TrOCR uses a Vision Transformer (ViT) as the encoder
to extract visual features, and a Transformer decoder to generate text autoregressively.
This architecture leverages the power of pre-trained language models.

## How It Works

Key features:

- Vision Transformer encoder for image understanding
- Transformer decoder with attention for text generation
- Autoregressive decoding with beam search
- Can leverage pre-trained models

Reference: Li et al., "TrOCR: Transformer-based Optical Character Recognition
with Pre-trained Models", AAAI 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrOCR(OCROptions<>)` | Creates a new TrOCR text recognizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `NumHeads` | Gets the number of attention heads in the transformer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameterCount` |  |
| `LoadWeightsAsync(String,CancellationToken)` |  |
| `LoadWeightsFromFile(String)` | Loads weights from a native TrOCR file format. |
| `Recognize(Tensor<>)` |  |
| `RecognizeText(Tensor<>)` |  |
| `SaveWeights(String)` |  |

