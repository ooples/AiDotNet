---
title: "VideoMAE<T>"
description: "Video Masked Autoencoder (VideoMAE) for video understanding and action recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.ActionRecognition`

Video Masked Autoencoder (VideoMAE) for video understanding and action recognition.

## For Beginners

VideoMAE is a self-supervised learning model for video understanding.
It learns powerful video representations by masking random patches in video frames
and training the model to reconstruct the missing content. This learned representation
can then be used for various tasks:

- Action recognition (identifying what's happening in a video)
- Video classification
- Temporal reasoning
- Video captioning

The key insight is that learning to reconstruct masked video teaches the model
about motion, appearance, and temporal patterns in videos.

## How It Works

**Technical Details:**

- Vision Transformer (ViT) architecture with temporal extension
- Tube masking strategy for spatiotemporal masking
- High masking ratio (75-90%) for efficient training
- Joint space-time attention mechanism

**Reference:** Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
NeurIPS 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VideoMAE` | Initializes a new instance of VideoMAE with default architecture (224x224, 400 classes). |
| `VideoMAE(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Int32,Double,VideoMAEOptions)` | Initializes a new instance of the VideoMAE class in native (trainable) mode. |
| `VideoMAE(NeuralNetworkArchitecture<>,String,Int32,Int32,VideoMAEOptions)` | Initializes a new instance of the VideoMAE class in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputHeight` | Gets the input height for frames. |
| `InputWidth` | Gets the input width for frames. |
| `MaskRatio` | Gets the masking ratio for pretraining. |
| `NumClasses` | Gets the number of action classes. |
| `NumFrames` | Gets the number of frames processed. |
| `SupportsTraining` | Gets whether training is supported. |
| `UseNativeMode` | Gets whether using native mode (trainable) or ONNX mode (inference only). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClassifyAction(Tensor<>)` | Classifies actions in a video clip. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` | Releases the unmanaged resources and optionally releases managed resources. |
| `ExtractFeatures(Tensor<>)` | Extracts video features for downstream tasks. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetTopKPredictions(Tensor<>,Int32)` | Gets the top-k predicted actions for a video. |
| `InitializeLayers` |  |
| `PoolTubelets(Tensor<>,Int32,Int32)` | Averages the per-tubelet feature rows that `Tensor{` folded into the batch dimension back down to one row per input video. |
| `PredictCore(Tensor<>)` |  |
| `PretrainMAE(Tensor<>)` | Performs masked autoencoder pretraining on a video. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

