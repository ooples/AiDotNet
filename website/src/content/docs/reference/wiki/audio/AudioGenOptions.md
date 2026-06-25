---
title: "AudioGenOptions"
description: "Configuration options for audio generation models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.AudioGen`

Configuration options for audio generation models.

## For Beginners

AudioGen works differently from TTS:

- TTS: Converts specific text to spoken words
- AudioGen: Creates sounds/music matching a description

Example prompts:

- "A dog barking in the distance"
- "Gentle piano music with rain sounds"
- "Crowd cheering at a sports event"

## How It Works

AudioGen models generate audio from text descriptions using a language model
approach with discrete audio codes (like EnCodec).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioGenOptions` | Initializes a new instance with default values. |
| `AudioGenOptions(AudioGenOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AudioCodecPath` | Gets or sets the path to the audio codec (decoder) model. |
| `Channels` | Gets or sets the number of audio channels (1=mono, 2=stereo). |
| `DurationSeconds` | Gets or sets the duration of generated audio in seconds. |
| `GuidanceScale` | Gets or sets the classifier-free guidance scale. |
| `LanguageModelPath` | Gets or sets the path to the language model. |
| `MaxDurationSeconds` | Gets or sets the maximum duration in seconds. |
| `ModelSize` | Gets or sets the model size to use. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `SampleRate` | Gets or sets the output sample rate. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `TextEncoderPath` | Gets or sets the path to the text encoder model. |
| `TopK` | Gets or sets the top-k value for sampling. |
| `TopP` | Gets or sets the top-p (nucleus) value for sampling. |

