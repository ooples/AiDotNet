---
title: "ECAPATDNNLanguageIdentifier<T>"
description: "ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network) for spoken language identification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.LanguageIdentification`

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network)
for spoken language identification.

## For Beginners

ECAPA-TDNN is like having a very sophisticated listener that can:

1. Hear patterns at different time scales (TDNN layers)
2. Focus on the most important sound characteristics (channel attention)
3. Combine information from multiple processing stages (MFA)
4. Handle audio of any length (attentive pooling)

This model is particularly good at:

- Identifying languages from short audio clips (3-10 seconds)
- Handling noisy or low-quality audio
- Distinguishing between similar languages (e.g., Spanish vs Portuguese)

Example usage:

## How It Works

ECAPA-TDNN is a state-of-the-art architecture originally designed for speaker verification
that has been adapted for language identification. It uses:

- Time Delay Neural Network (TDNN) layers with dilated convolutions
- Squeeze-Excitation (SE) blocks for channel attention
- Multi-layer feature aggregation (MFA) for combining information across layers
- Attentive statistics pooling for variable-length utterances

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ECAPATDNNLanguageIdentifier(NeuralNetworkArchitecture<>,IReadOnlyList<String>,ECAPATDNNOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates an ECAPA-TDNN language identifier for native training. |
| `ECAPATDNNLanguageIdentifier(NeuralNetworkArchitecture<>,String,ECAPATDNNOptions)` | Creates an ECAPA-TDNN language identifier with ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension produced by this model. |
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

