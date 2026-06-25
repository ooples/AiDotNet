---
title: "ISpeechRecognizer<T>"
description: "Interface for speech recognition models that transcribe audio to text (ASR - Automatic Speech Recognition)."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for speech recognition models that transcribe audio to text (ASR - Automatic Speech Recognition).

## For Beginners

Speech recognition is like having a transcriptionist listen to audio
and type out what they hear.

How speech recognition works:

1. Audio is converted to features (spectrograms or mel-spectrograms)
2. The model processes these features to identify speech patterns
3. Patterns are decoded into words and sentences

Common use cases:

- Voice assistants (Siri, Alexa, Google Assistant)
- Video/podcast transcription
- Real-time captioning for accessibility
- Voice typing and dictation

Key challenges:

- Different accents and speaking styles
- Background noise and multiple speakers
- Domain-specific vocabulary (medical, legal terms)

## How It Works

Speech recognition models convert spoken audio into written text. They analyze audio waveforms
or spectrograms to identify phonemes, words, and sentences. Modern speech recognition uses
encoder-decoder architectures (like Whisper) or CTC-based models.

This interface extends `IFullModel` for Tensor-based audio processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `SampleRate` | Gets the sample rate expected by this model. |
| `SupportedLanguages` | Gets the list of languages supported by this model. |
| `SupportsStreaming` | Gets whether this model supports real-time streaming transcription. |
| `SupportsWordTimestamps` | Gets whether this model can identify timestamps for each word. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectLanguage(Tensor<>)` | Detects the language spoken in the audio. |
| `DetectLanguageProbabilities(Tensor<>)` | Gets language detection probabilities for the audio. |
| `StartStreamingSession(String)` | Starts a streaming transcription session. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio to text. |
| `TranscribeAsync(Tensor<>,String,Boolean,CancellationToken)` | Transcribes audio to text asynchronously. |

