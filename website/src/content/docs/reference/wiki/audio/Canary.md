---
title: "Canary<T>"
description: "Canary multilingual speech recognition and translation model from NVIDIA."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

Canary multilingual speech recognition and translation model from NVIDIA.

## For Beginners

Canary is like a multilingual transcription assistant. It can listen
to speech in many languages and either transcribe it (write down what was said) or translate
it into another language, all with a single model. You tell it what you want via a prompt.

**Usage:**

## How It Works

Canary (NVIDIA, 2024) is a multilingual ASR/ST model based on the Fast Conformer encoder
with a multi-task decoder. It supports transcription and translation across many languages
using a single unified architecture with task-specific prompting, achieving strong WER
scores across English, German, Spanish, and French.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Canary(NeuralNetworkArchitecture<>,CanaryOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Canary model in native training mode. |
| `Canary(NeuralNetworkArchitecture<>,String,CanaryOptions)` | Creates a Canary model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` |  |
| `SampleRate` |  |
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

