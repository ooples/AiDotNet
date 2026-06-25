---
title: "YuEOptions"
description: "Configuration options for the YuE music generation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the YuE music generation model.

## For Beginners

YuE is like having a virtual band that can write and perform an
entire song. You give it lyrics and a style ("pop, female vocalist, upbeat") and it generates
a complete song with singing, instruments, and production. Unlike most AI music tools that
only make short clips, YuE can create full-length songs.

## How It Works

YuE (Yuan et al., 2025) is a full-song music generation model that generates complete songs
with vocals and accompaniment from lyrics and genre/style tags. It uses a dual-AR architecture
where a lyrics-conditioned language model generates semantic tokens and a second stage
produces acoustic tokens, generating songs of several minutes in length.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcousticCodebookSize` | Gets or sets the acoustic codec codebook size. |
| `AcousticDim` | Gets or sets the acoustic stage model dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `LyricsVocabSize` | Gets or sets the lyrics token vocabulary size. |
| `MaxDurationSeconds` | Gets or sets the maximum generation duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumAcousticHeads` | Gets or sets the number of acoustic stage attention heads. |
| `NumAcousticLayers` | Gets or sets the number of acoustic stage transformer layers. |
| `NumAcousticQuantizers` | Gets or sets the number of acoustic codec quantizers. |
| `NumSemanticHeads` | Gets or sets the number of semantic stage attention heads. |
| `NumSemanticLayers` | Gets or sets the number of semantic stage transformer layers. |
| `NumStyleTags` | Gets or sets the number of genre/style tag embeddings. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `RepetitionPenalty` | Gets or sets the repetition penalty factor. |
| `SampleRate` | Gets or sets the output audio sample rate in Hz. |
| `SemanticDim` | Gets or sets the semantic stage model dimension. |
| `SemanticVocabSize` | Gets or sets the semantic token vocabulary size. |
| `StyleEmbeddingDim` | Gets or sets the style embedding dimension. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |

