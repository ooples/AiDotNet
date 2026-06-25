---
title: "OpenVocabSegmentationBase<T>"
description: "Abstract base class for open-vocabulary segmentation models that segment objects described by arbitrary text without being limited to a fixed class set."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for open-vocabulary segmentation models that segment objects described
by arbitrary text without being limited to a fixed class set.

## For Beginners

Traditional segmentation models only recognize objects they were trained
on (e.g., "car", "person"). Open-vocabulary models use language understanding (typically CLIP)
to segment anything you describe in text, even novel concepts.

Models extending this base class: SAN, CAT-Seg, SED, Open-Vocabulary SAM, Grounded SAM 2, Mask-Adapter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenVocabSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Int32)` | Initializes the base in native (trainable) mode. |
| `OpenVocabSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxCategories` |  |
| `MaxPromptLength` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentWithPrompt(Tensor<>,String)` |  |
| `SegmentWithText(Tensor<>,IReadOnlyList<String>)` |  |

