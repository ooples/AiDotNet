---
title: "CanaryOptions"
description: "Configuration options for the Canary model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SpeechRecognition`

Configuration options for the Canary model.

## For Beginners

Canary is like a multilingual transcription assistant. It can listen
to speech in many languages and either transcribe it (write down what was said) or translate
it into another language, all with a single model.

## How It Works

Canary (NVIDIA, 2024) is a multilingual speech recognition model based on the Fast Conformer
encoder with a multi-task decoder. It supports automatic speech recognition (ASR), speech
translation (ST), and language identification across many languages, using a single unified
architecture with task-specific prompting.

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamWidth` | Gets or sets the beam width for decoding. |
| `DecoderDim` | Gets or sets the decoder dimension. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderDim` | Gets or sets the encoder dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxOutputTokens` | Gets or sets the maximum output tokens. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SubsamplingFactor` | Gets or sets the subsampling factor. |
| `SupportedLanguages` | Gets or sets the supported languages. |
| `TargetLanguage` | Gets or sets the target language for translation. |
| `Variant` | Gets or sets the model variant. |
| `VocabSize` | Gets or sets the vocabulary size. |

