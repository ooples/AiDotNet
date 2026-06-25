---
title: "Conformer<T>"
description: "Conformer speech recognition model (Gulati et al., 2020, Google)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

Conformer speech recognition model (Gulati et al., 2020, Google).

## For Beginners

The Conformer is like a super-powered speech encoder.
Before it, speech models used either:

- Transformers (good at seeing the whole sentence, bad at fine details)
- CNNs (good at local patterns like individual sounds, bad at context)

The Conformer uses BOTH in each layer, so it can hear the "s" at the end of a
word (local detail) while also understanding the sentence structure (global context).

**Usage:**

## How It Works

The Conformer combines convolution and self-attention in a novel macaron-style
architecture: Feed-Forward / Self-Attention / Convolution / Feed-Forward. This captures
both local (phoneme-level) and global (sentence-level) dependencies. Conformer-CTC
achieves WER 1.9%/3.9% on LibriSpeech test-clean/other and is the backbone of
production ASR at Google, NVIDIA NeMo, and many other systems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Conformer(NeuralNetworkArchitecture<>,ConformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Conformer model in native training mode. |
| `Conformer(NeuralNetworkArchitecture<>,String,ConformerOptions)` | Creates a Conformer model in ONNX inference mode. |

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

