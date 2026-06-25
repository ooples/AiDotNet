---
title: "RNNTransducer<T>"
description: "RNN-Transducer (RNN-T) streaming speech recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

RNN-Transducer (RNN-T) streaming speech recognition model.

## For Beginners

RNN-T is a real-time speech recognizer ideal for live transcription.
Unlike batch models (like Whisper) that need the whole audio, RNN-T processes speech as
it arrives - perfect for live captioning and voice assistants.

It has three parts:

1. **Encoder** - Listens to the audio (converts sound to features)
2. **Predictor** - Remembers what was already said (like a small language model)
3. **Joiner** - Combines both to decide the next output token

Think of it like a court stenographer who listens (encoder), remembers the context
(predictor), and types the next word (joiner) - all in real time.

**Usage:**

## How It Works

RNN-Transducer (Graves, 2012; He et al., 2019) combines an audio encoder with a label
prediction network and a joint network to produce a streaming ASR system. Unlike CTC,
RNN-T can model output dependencies through its prediction network, achieving strong
results on LibriSpeech (WER 2.0% with LM) without an external language model. It is
the backbone of on-device ASR in Google's Pixel phones and NVIDIA Riva.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RNNTransducer(NeuralNetworkArchitecture<>,RNNTransducerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an RNN-T model in native training mode. |
| `RNNTransducer(NeuralNetworkArchitecture<>,String,RNNTransducerOptions)` | Creates an RNN-T model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedLanguages` |  |
| `SupportsStreaming` |  |
| `SupportsWordTimestamps` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectLanguage(Tensor<>)` |  |
| `DetectLanguageProbabilities(Tensor<>)` |  |
| `StartStreamingSession(String)` |  |
| `Transcribe(Tensor<>,String,Boolean)` |  |
| `TranscribeAsync(Tensor<>,String,Boolean,CancellationToken)` |  |

