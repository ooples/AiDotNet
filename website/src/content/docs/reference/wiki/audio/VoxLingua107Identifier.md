---
title: "VoxLingua107Identifier<T>"
description: "VoxLingua107 language identifier supporting 107 languages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.LanguageIdentification`

VoxLingua107 language identifier supporting 107 languages.

## For Beginners

VoxLingua107 is like having a polyglot friend who can
recognize 107 different languages just by listening.

Key features:

- Covers most of the world's major languages
- Trained on real-world YouTube audio (diverse accents and recording conditions)
- Can identify languages even from short clips (3-10 seconds)
- Handles code-switching and multilingual speakers

Example usage:

## How It Works

VoxLingua107 is a language identification model trained on the VoxLingua107 dataset,
which contains speech samples from 107 languages collected from YouTube videos.
The model uses the ECAPA-TDNN architecture and is specifically optimized for
large-scale multilingual language identification.

Supported language families include:

- Indo-European (English, Spanish, French, German, Russian, Hindi, etc.)
- Sino-Tibetan (Mandarin, Cantonese, etc.)
- Afro-Asiatic (Arabic, Hebrew, Amharic, etc.)
- Austronesian (Indonesian, Tagalog, Malay, etc.)
- Niger-Congo (Swahili, Yoruba, Zulu, etc.)
- Altaic (Turkish, Korean, Japanese, Mongolian, etc.)
- And many more...

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoxLingua107Identifier(NeuralNetworkArchitecture<>,String,VoxLingua107Options)` | Creates a VoxLingua107 identifier with ONNX model for inference. |
| `VoxLingua107Identifier(NeuralNetworkArchitecture<>,VoxLingua107Options,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Creates a VoxLingua107 identifier for native training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `NumLanguages` | Gets the number of supported languages (107). |
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
| `GetLanguagesByFamily(String)` | Gets all languages in a specific language family. |
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

## Fields

| Field | Summary |
|:-----|:--------|
| `VoxLingua107Languages` | The 107 language codes supported by VoxLingua107 (ISO 639-1/3). |

