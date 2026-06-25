---
title: "WhisperOptions"
description: "Configuration options for the Whisper speech recognition model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Whisper`

Configuration options for the Whisper speech recognition model.

## For Beginners

Whisper comes in different sizes (tiny to large).
Smaller models are faster but less accurate. Larger models are more accurate but slower.

- **Tiny**: ~39M parameters, fastest, good for quick transcription
- **Base**: ~74M parameters, balanced speed/accuracy
- **Small**: ~244M parameters, good accuracy
- **Medium**: ~769M parameters, high accuracy
- **Large**: ~1.5B parameters, best accuracy, slow

## How It Works

Whisper is a speech recognition model developed by OpenAI that can
transcribe audio in multiple languages and perform translation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WhisperOptions` | Initializes a new instance with default values. |
| `WhisperOptions(WhisperOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamSize` | Gets or sets the beam size for beam search decoding. |
| `DecoderModelPath` | Gets or sets the path to the decoder ONNX model. |
| `EncoderModelPath` | Gets or sets the path to the encoder ONNX model. |
| `Language` | Gets or sets the language code for transcription (e.g., "en", "es", "fr"). |
| `MaxAudioLengthSeconds` | Gets or sets the maximum length of audio to process in seconds. |
| `MaxTokens` | Gets or sets the maximum number of tokens to generate. |
| `ModelSize` | Gets or sets the model size to use. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX execution options. |
| `ReturnTimestamps` | Gets or sets whether to return timestamps with the transcription. |
| `SampleRate` | Gets or sets the sample rate expected by the model. |
| `Temperature` | Gets or sets the temperature for sampling. |
| `Translate` | Gets or sets whether to translate to English. |
| `WordTimestamps` | Gets or sets whether to include word-level timestamps. |

