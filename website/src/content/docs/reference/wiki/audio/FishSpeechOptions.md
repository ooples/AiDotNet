---
title: "FishSpeechOptions"
description: "Configuration options for the Fish Speech TTS model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the Fish Speech TTS model.

## For Beginners

Fish Speech is a fast, open-source text-to-speech system. Give it
a few seconds of someone's voice and some text, and it will speak the text in that person's
voice. It works in many languages and is fast enough for live conversations. Think of it
as an open-source alternative to commercial voice cloning services.

## How It Works

Fish Speech (Fish Audio, 2024) is an open-source multilingual TTS system that uses a
dual-AR architecture with grouped finite scalar quantization (GFSQ). It supports zero-shot
voice cloning from a few seconds of reference audio and generates natural-sounding speech
in multiple languages with very low latency suitable for real-time streaming.

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets or sets the GFSQ codebook size. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxDurationSeconds` | Gets or sets the maximum generation duration in seconds. |
| `MinReferenceSeconds` | Gets or sets the minimum reference audio duration in seconds for voice cloning. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumGroups` | Gets or sets the number of GFSQ groups. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `NumSemanticHeads` | Gets or sets the number of semantic attention heads. |
| `NumSemanticLayers` | Gets or sets the number of semantic transformer layers. |
| `NumVocoderLayers` | Gets or sets the number of VQGAN vocoder layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `RepetitionPenalty` | Gets or sets the repetition penalty factor. |
| `SampleRate` | Gets or sets the output audio sample rate in Hz. |
| `SemanticDim` | Gets or sets the semantic language model dimension. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `TextVocabSize` | Gets or sets the text token vocabulary size. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |
| `VocoderDim` | Gets or sets the VQGAN vocoder hidden dimension. |

