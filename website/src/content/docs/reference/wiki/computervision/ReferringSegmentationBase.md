---
title: "ReferringSegmentationBase<T>"
description: "Abstract base class for referring segmentation models that segment objects from natural language descriptions with complex reasoning about spatial relationships and attributes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Segmentation.Common`

Abstract base class for referring segmentation models that segment objects from natural language
descriptions with complex reasoning about spatial relationships and attributes.

## For Beginners

Referring segmentation goes beyond open-vocabulary by understanding
complex descriptions like "the person standing behind the counter" or "the animal that could
be dangerous". These models typically combine a large language model (LLM) with a segmentation
backbone.

Models extending this base class: LISA, VideoLISA, GLaMM, OMG-LLaVA, PixelLM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReferringSegmentationBase(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32)` | Initializes the base in native (trainable) mode. |
| `ReferringSegmentationBase(NeuralNetworkArchitecture<>,String,Int32,Int32)` | Initializes the base in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxTextLength` |  |
| `SupportsConversation` |  |
| `SupportsVideoInput` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SegmentFromConversation(Tensor<>,IReadOnlyList<ValueTuple<String,String>>,String)` |  |
| `SegmentFromConversationInternal(Tensor<>,IReadOnlyList<ValueTuple<String,String>>,String)` | Model-specific conversational segmentation. |
| `SegmentFromExpression(Tensor<>,String)` |  |
| `SegmentVideoFromExpression(Tensor<>,String)` |  |
| `SegmentVideoFromExpressionInternal(Tensor<>,String)` | Model-specific video referring segmentation. |

