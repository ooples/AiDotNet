---
title: "SpeakerEmbeddingExtractor<T>"
description: "Extracts speaker embeddings (d-vectors) from audio for speaker recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

Extracts speaker embeddings (d-vectors) from audio for speaker recognition.

## For Beginners

Each person's voice has unique characteristics
like pitch, rhythm, and timbre (tone color). This class converts audio into
a numerical "fingerprint" of the speaker's voice.

These embeddings are vectors (lists of numbers) that are:

- Close together for the same speaker
- Far apart for different speakers

Usage (ONNX Mode):

Usage (Native Training Mode):

## How It Works

Speaker embeddings are compact vector representations that capture the
unique characteristics of a speaker's voice. These can be used for
speaker verification (is this the same person?) and speaker identification
(who is speaking?).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpeakerEmbeddingExtractor` | Creates a SpeakerEmbeddingExtractor with default settings for native training mode. |
| `SpeakerEmbeddingExtractor(NeuralNetworkArchitecture<>,Int32,Int32,Double,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a SpeakerEmbeddingExtractor for native training mode. |
| `SpeakerEmbeddingExtractor(NeuralNetworkArchitecture<>,String,Int32,Int32,Double,OnnxModelOptions)` | Creates a SpeakerEmbeddingExtractor for ONNX inference with a pretrained model. |
| `SpeakerEmbeddingExtractor(SpeakerEmbeddingOptions)` | Creates a SpeakerEmbeddingExtractor with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasNeuralModel` | Gets whether a neural model is loaded. |
| `IsOnnxMode` | Gets whether the model is in ONNX inference mode. |
| `MinimumDurationSeconds` | Gets the minimum audio duration required for reliable embedding extraction. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#ISpeakerEmbeddingExtractor{T}#AggregateEmbeddings(IReadOnlyList<Tensor<>>)` | Aggregates multiple embeddings into a single representative embedding. |
| `AiDotNet#Interfaces#ISpeakerEmbeddingExtractor{T}#NormalizeEmbedding(Tensor<>)` | Normalizes an embedding to unit length. |
| `ComputeSimilarity(SpeakerEmbedding<>,SpeakerEmbedding<>)` | Computes cosine similarity between two speaker embeddings (legacy API). |
| `ComputeSimilarity(Tensor<>,Tensor<>)` | Computes similarity between two speaker embeddings. |
| `CreateNewInstance` | Creates a new instance of this model for cloning. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data. |
| `Dispose(Boolean)` | Disposes the model and releases resources. |
| `Extract(Tensor<>)` | Extracts a speaker embedding from audio (legacy API). |
| `Extract(Vector<>)` | Extracts a speaker embedding from audio (legacy API). |
| `ExtractBatch(IEnumerable<Tensor<>>)` | Extracts embeddings from multiple audio segments (legacy API). |
| `ExtractEmbedding(Tensor<>)` | Extracts a speaker embedding from audio. |
| `ExtractEmbeddingAsync(Tensor<>,CancellationToken)` | Extracts a speaker embedding from audio asynchronously. |
| `ExtractEmbeddings(IReadOnlyList<Tensor<>>)` | Extracts embeddings from multiple audio segments. |
| `ExtractTensor(Tensor<>)` | Extracts speaker embedding from audio as a Tensor. |
| `Forward(Tensor<>)` | X-Vector forward pass: run the frame-level TDNN/attention stack, pool across the time dimension, then run the segment-level FC layers and embedding projection. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` | Gets metadata about the model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers for the speaker embedding model. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into the final result format. |
| `PredictCore(Tensor<>)` | Makes a prediction using the model. |
| `PreprocessAudio(Tensor<>)` | Preprocesses raw audio for model input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on input data. |
| `UpdateParameters(Vector<>)` | Updates model parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `SegmentLevelTrailingLayers` | Number of trailing layers that operate on the segment-level (post-pooling) representation. |
| `_disposed` | Whether the model has been disposed. |
| `_hiddenDim` | Hidden dimension for encoder layers. |
| `_lossFunction` | Loss function for training. |
| `_modelPath` | Path to the speaker model ONNX file. |
| `_numEncoderLayers` | Number of encoder layers. |
| `_numHeads` | Number of attention heads. |
| `_onnxModel` | ONNX speaker embedding model. |
| `_optimizer` | Optimizer for training. |
| `_options` | Speaker embedding options. |
| `_useNativeMode` | Whether the model is operating in native training mode. |

