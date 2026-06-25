---
title: "OCRBase<T>"
description: "Base class for OCR models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.OCR`

Base class for OCR models.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OCRBase(OCROptions<>)` | Creates a new OCR model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Name` | Name of this OCR model. |
| `VocabularySize` | Gets the vocabulary size (number of classes). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeConfidence(Tensor<>,String)` | Computes confidence from logits. |
| `DecodeAttention(Tensor<>,Int32)` | Decodes attention-based output to text. |
| `DecodeCTC(Tensor<>)` | Decodes CTC output to text. |
| `DeepCopy` |  |
| `GetParameterCount` | Gets the total parameter count. |
| `GetParameters` |  |
| `LoadWeightsAsync(String,CancellationToken)` | Loads pretrained weights. |
| `Predict(Tensor<>)` | Runs OCR and returns region info as a tensor [numRegions, 6]. |
| `PreprocessCrop(Tensor<>)` | Preprocesses a text crop for recognition. |
| `Recognize(Tensor<>)` | Recognizes text in an image. |
| `RecognizeText(Tensor<>)` | Recognizes text in a cropped text region. |
| `ResizeBilinear(Tensor<>,Int32,Int32)` | Resizes tensor using bilinear interpolation. |
| `SaveWeights(String)` | Saves model weights. |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CharToIndex` | Character to index mapping. |
| `DefaultCharacterSet` | Default character set for recognition. |
| `IndexToChar` | Index to character mapping. |

