---
title: "IVideoModel<T>"
description: "Base interface for all video AI models in AiDotNet."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Video.Interfaces`

Base interface for all video AI models in AiDotNet.

## For Beginners

A video AI model processes video data (sequences of frames)
to perform tasks like enhancement, classification, or feature extraction.

Key concepts:

- Video tensors have shape [batch, numFrames, channels, height, width]
- Models can run in Native mode (pure C#) or ONNX mode (optimized runtime)
- All models support both training and inference
- Models inherit full serialization, checkpointing, and gradient computation from IFullModel

Example usage:

## How It Works

This interface extends IFullModel to provide the core contract for video AI models,
inheriting standard methods for training, inference, model persistence, and gradient computation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedInputShape` | Gets the expected input tensor shape for this model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelSummary` | Gets a summary of the model architecture. |
| `PredictBatch(IEnumerable<Tensor<>>)` | Processes multiple videos in a batch. |
| `TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<TrainingProgress>,CancellationToken)` | Trains the model on video data asynchronously with progress reporting. |
| `ValidateInputShape(Tensor<>)` | Validates that an input tensor has the correct shape for this model. |

