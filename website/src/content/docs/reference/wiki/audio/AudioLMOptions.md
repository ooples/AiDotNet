---
title: "AudioLMOptions"
description: "Configuration options for the AudioLM audio generation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the AudioLM audio generation model.

## For Beginners

AudioLM generates natural-sounding audio by "thinking" about
what to say/play at two levels: first the big-picture meaning (semantic), then the
fine details of how it sounds (acoustic). This two-stage approach produces audio that
is both coherent and high-fidelity, like a writer who first outlines a story then
adds the vivid details.

## How It Works

AudioLM (Borsos et al., 2023, Google) generates high-quality, coherent audio by
combining semantic tokens (from a self-supervised model like w2v-BERT) with acoustic
tokens (from a neural codec like SoundStream). A hierarchical language model generates
semantic tokens first for high-level structure, then acoustic tokens for fine detail.

## Properties

| Property | Summary |
|:-----|:--------|
| `CoarseCodebookSize` | Gets or sets the coarse acoustic codebook size. |
| `CoarseDim` | Gets or sets the coarse model dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FineCodebookSize` | Gets or sets the fine acoustic codebook size. |
| `FineDim` | Gets or sets the fine model dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxDurationSeconds` | Gets or sets the maximum generation duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumCoarseLayers` | Gets or sets the number of coarse transformer layers. |
| `NumCoarseQuantizers` | Gets or sets the number of coarse quantizers. |
| `NumFineLayers` | Gets or sets the number of fine transformer layers. |
| `NumFineQuantizers` | Gets or sets the total number of fine quantizers. |
| `NumSemanticHeads` | Gets or sets the number of semantic attention heads. |
| `NumSemanticLayers` | Gets or sets the number of semantic transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SemanticDim` | Gets or sets the semantic model dimension. |
| `SemanticFrameRate` | Gets or sets the semantic token frame rate (tokens/second). |
| `SemanticVocabSize` | Gets or sets the semantic token vocabulary size (from w2v-BERT). |
| `Temperature` | Gets or sets the temperature for sampling. |
| `TopK` | Gets or sets the top-k sampling parameter. |

