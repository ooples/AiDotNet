---
title: "OMGLLaVA<T>"
description: "OMG-LLaVA: Bridging Image-Level, Object-Level, and Pixel-Level understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Referring`

OMG-LLaVA: Bridging Image-Level, Object-Level, and Pixel-Level understanding.

## For Beginners

Unified visual perception and understanding. Object-level and pixel-level reasoning.

Common use cases:

- Unified visual perception and understanding
- Object-level and pixel-level reasoning
- Visual question answering with segmentation
- Multi-granularity visual understanding

## How It Works

**Technical Details:**

- Universal segmentation model (Mask2Former) + LLM (LLaVA)
- Bridges perception (segmentation) and understanding (reasoning)
- Object-level queries from segmentation feed into LLM
- Pixel-level grounding from LLM responses

**Reference:** Zhang et al., "OMG-LLaVA: Bridging Image-Level, Object-Level, Pixel-Level Reasoning and Understanding", arXiv 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OMGLLaVA(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Double,OMGLLaVAOptions)` | Initializes OMGLLaVA in native (trainable) mode. |
| `OMGLLaVA(NeuralNetworkArchitecture<>,String,Int32,OMGLLaVAOptions)` | Initializes OMGLLaVA in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this OMGLLaVA instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this model's configuration. |
| `InitializeLayers` | Initializes the encoder and decoder layers. |
| `PredictCore(Tensor<>)` | Runs a forward pass to produce segmentation logits. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

