---
title: "VALLEOptions"
description: "Configuration options for the VALL-E zero-shot TTS model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the VALL-E zero-shot TTS model.

## For Beginners

VALL-E can hear someone speak for 3 seconds and then generate
new speech that sounds just like them. It works by converting speech into "audio words"
(codec tokens) and then using a language model - the same kind of AI behind ChatGPT -
to predict what tokens come next, given a text prompt and the speaker's voice sample.

## How It Works

VALL-E (Wang et al., 2023, Microsoft) treats text-to-speech as a language modeling problem
using discrete audio codes from EnCodec. A 3-second enrollment recording is enough to
synthesize speech in the speaker's voice. It uses an autoregressive (AR) model for the
first codebook layer and a non-autoregressive (NAR) model for remaining layers.

## Properties

| Property | Summary |
|:-----|:--------|
| `ARHiddenDim` | Gets or sets the AR model hidden dimension. |
| `CodebookSize` | Gets or sets the codec codebook size (from EnCodec). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaxDurationSeconds` | Gets or sets the maximum generation duration in seconds. |
| `MinEnrollmentSeconds` | Gets or sets the minimum enrollment audio duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NARHiddenDim` | Gets or sets the NAR model hidden dimension. |
| `NumARHeads` | Gets or sets the number of AR attention heads. |
| `NumARLayers` | Gets or sets the number of AR transformer layers. |
| `NumCodebooks` | Gets or sets the number of codec quantizer layers (8 for EnCodec). |
| `NumNARHeads` | Gets or sets the number of NAR attention heads. |
| `NumNARLayers` | Gets or sets the number of NAR transformer layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `PhonemeVocabSize` | Gets or sets the phoneme vocabulary size. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Temperature` | Gets or sets the temperature for AR sampling. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |

