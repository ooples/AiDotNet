---
title: "HTSAT<T>"
description: "HTS-AT (Hierarchical Token-Semantic Audio Transformer) model for efficient audio classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

HTS-AT (Hierarchical Token-Semantic Audio Transformer) model for efficient audio classification.

## For Beginners

HTS-AT is like reading a book at different zoom levels. First it
reads individual words (small patches), then sentences (merged patches), then paragraphs
(further merged), and finally understands the whole story. At each level, it uses "window
attention" - looking at nearby patches first, which is much faster than looking at everything.

**Usage with ONNX (recommended):**

## How It Works

HTS-AT (Chen et al., ICASSP 2022) is a hierarchical Transformer architecture that uses
Swin Transformer blocks with a token-semantic module for efficient audio classification.
It achieves 47.1% mAP on AudioSet-2M with only 30M parameters, making it more parameter-efficient
than models like AST (87M parameters) while achieving higher accuracy.

**Architecture:** HTS-AT processes audio spectrograms hierarchically through four stages:

- **Patch embedding**: 4x4 patch embedding of the mel spectrogram
- **Stage 1**: 2 Swin blocks at 96-dim with local window attention
- **Stage 2**: 2 Swin blocks at 192-dim (patch merging doubles channels, halves resolution)
- **Stage 3**: 6 Swin blocks at 384-dim (the deepest processing stage)
- **Stage 4**: 2 Swin blocks at 768-dim
- **Token-semantic module**: Groups tokens by semantic meaning for global context
- **Classification head**: Projects to class logits with sigmoid for multi-label

**References:**

- Paper: "HTS-AT: A Hierarchical Token-Semantic Audio Transformer" (Chen et al., ICASSP 2022)
- Repository: https://github.com/RetroCirce/HTS-Audio-Transformer

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HTSAT(NeuralNetworkArchitecture<>,HTSATOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an HTS-AT model for native training mode. |
| `HTSAT(NeuralNetworkArchitecture<>,String,HTSATOptions)` | Creates an HTS-AT model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` |  |
| `SupportedEvents` |  |
| `TimeResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(HTSATOptions,IProgress<Double>,CancellationToken)` | Creates an HTS-AT model asynchronously with optional model download. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,)` |  |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `Dispose(Boolean)` |  |
| `GetEventProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels. |

