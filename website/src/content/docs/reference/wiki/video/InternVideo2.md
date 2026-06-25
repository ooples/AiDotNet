---
title: "InternVideo2<T>"
description: "InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Understanding`

InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding.

## For Beginners

InternVideo2 understands video content by analyzing frames
and learning relationships between visual content and language. It can:

- Classify videos (what's happening?)
- Find videos matching text descriptions
- Answer questions about video content
- Generate video captions

Example usage (native mode for training):

Example usage (ONNX mode for inference only):

## How It Works

InternVideo2 is a state-of-the-art video understanding model that combines:

- Video-text contrastive learning
- Masked video modeling
- Video-text generative learning

**Reference:** "InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding"
https://arxiv.org/abs/2403.15377

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InternVideo2` | Creates an InternVideo2 model using native layers for training and inference. |
| `InternVideo2(NeuralNetworkArchitecture<>,String,Int32,InternVideo2Options)` | Creates an InternVideo2 model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbedDim` | Gets the embedding dimension. |
| `NumFrames` | Gets the number of frames processed. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two tensors. |
| `ComputeSimilarity(Tensor<>,Tensor<>)` | Computes similarity between video and text embeddings. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncodeVideo(Tensor<>)` | Encodes video frames into an embedding vector. |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embedDim` | Embedding dimension for the model. |
| `_imageSize` | Input image size. |
| `_lossFunction` | The loss function for training. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numFrames` | Number of video frames to process. |
| `_numHeads` | Number of attention heads. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_optimizer` | The optimizer used for training. |
| `_patchSize` | Patch size for tokenization. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

