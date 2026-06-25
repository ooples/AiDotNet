---
title: "MusicGenOptions"
description: "Configuration options for MusicGen text-to-music generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicGen`

Configuration options for MusicGen text-to-music generation.

## For Beginners

MusicGen generates actual music from descriptions:

Example prompts:

- "Upbeat electronic dance music with heavy bass"
- "Calm acoustic guitar melody with soft drums"
- "Epic orchestral piece with dramatic strings"
- "Lo-fi hip hop beats for studying"

Tips for good prompts:

- Be specific about genre, instruments, and mood
- Include tempo hints (fast, slow, moderate)
- Mention energy level (energetic, calm, building)

## How It Works

MusicGen is Meta's state-of-the-art music generation model that creates
high-quality music from text descriptions. It uses a single-stage transformer
language model operating over EnCodec audio codes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MusicGenOptions` | Initializes a new instance with default values. |
| `MusicGenOptions(MusicGenOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets or sets the codebook vocabulary size. |
| `DropoutRate` | Gets or sets the dropout rate for training. |
| `DurationSeconds` | Gets or sets the default duration of generated music in seconds. |
| `EnCodecDecoderPath` | Gets or sets the path to the EnCodec decoder ONNX model. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LanguageModelPath` | Gets or sets the path to the language model ONNX model. |
| `MaxDurationSeconds` | Gets or sets the maximum duration in seconds. |
| `MaxTextLength` | Gets or sets the maximum text sequence length. |
| `ModelSize` | Gets or sets the model size variant. |
| `NumCodebooks` | Gets or sets the number of EnCodec codebooks to use. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `SampleRate` | Gets or sets the output sample rate in Hz. |
| `Stereo` | Gets or sets whether to generate stereo audio. |
| `Temperature` | Gets or sets the sampling temperature. |
| `TextEncoderPath` | Gets or sets the path to the text encoder ONNX model. |
| `TopK` | Gets or sets the top-k sampling parameter. |
| `TopP` | Gets or sets the top-p (nucleus) sampling parameter. |
| `UseDelayPattern` | Gets or sets whether to use the delay pattern. |

