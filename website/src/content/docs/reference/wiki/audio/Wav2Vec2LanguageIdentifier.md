---
title: "Wav2Vec2LanguageIdentifier<T>"
description: "Wav2Vec2 model fine-tuned for spoken language identification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.LanguageIdentification`

Wav2Vec2 model fine-tuned for spoken language identification.

## For Beginners

Wav2Vec2 is like a very attentive listener that:

1. First breaks down the raw sound wave into small pieces (feature encoder)
2. Then looks at how all these pieces relate to each other (transformer)
3. Finally makes a decision about what language is being spoken (classifier)

Key advantages:

- Works directly on raw audio (no need for handcrafted features like MFCCs)
- Pre-trained on massive amounts of unlabeled speech data
- Can recognize languages even with limited labeled training data

Example usage:

## How It Works

Wav2Vec2 is Meta's self-supervised speech representation learning model that learns
powerful representations directly from raw audio waveforms. When fine-tuned for
language identification, it achieves state-of-the-art performance on many benchmarks.

Architecture overview:

- Feature Encoder: 7 temporal convolution layers that process raw waveform
- Transformer Encoder: 12-24 transformer blocks for contextual representations
- Classification Head: Linear projection to language classes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Wav2Vec2LanguageIdentifier(NeuralNetworkArchitecture<>,IReadOnlyList<String>,Wav2Vec2LidOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a Wav2Vec2 language identifier for native training. |
| `Wav2Vec2LanguageIdentifier(NeuralNetworkArchitecture<>,String,Wav2Vec2LidOptions)` | Creates a Wav2Vec2 language identifier with ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenSize` | Gets the hidden size of the transformer. |
| `SupportedLanguages` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AreSameLanguage(Tensor<>,Tensor<>)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetLanguageDisplayName(String)` |  |
| `GetLanguageProbabilities(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetTopLanguages(Tensor<>,Int32)` |  |
| `IdentifyLanguage(Tensor<>)` |  |
| `IdentifyLanguageSegments(Tensor<>,Int32)` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessAudio(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

