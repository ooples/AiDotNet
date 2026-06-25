---
title: "VoiceCraftOptions"
description: "Configuration options for the VoiceCraft speech editing and generation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the VoiceCraft speech editing and generation model.

## For Beginners

VoiceCraft can do two amazing things: (1) edit speech like you
edit text - change specific words in a recording while keeping the speaker's voice, and
(2) clone a voice from a few seconds of audio and generate new speech. It's like having
a "find and replace" for spoken words, plus a voice cloning tool.

## How It Works

VoiceCraft (Peng et al., 2024) is a neural codec language model for speech editing and
zero-shot TTS. It uses a token rearrangement procedure with causal masking that enables
both editing existing speech (replacing/inserting words) and generating new speech from
a short prompt, achieving high naturalness and speaker similarity.

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets or sets the codec codebook size. |
| `CodecEmbeddingDim` | Gets or sets the codec embedding dimension per quantizer. |
| `CodecFrameRate` | Gets or sets the codec frame rate (frames per second). |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EditContextSeconds` | Gets or sets the context window size in seconds for speech editing. |
| `HiddenDim` | Gets or sets the model hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `MaskRatio` | Gets or sets the masking ratio for causal masking during editing. |
| `MaxDurationSeconds` | Gets or sets the maximum generation duration in seconds. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumMels` | Gets or sets the number of mel spectrogram channels. |
| `NumQuantizers` | Gets or sets the number of codec quantizers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |

