---
title: "CTCDecoder<T>"
description: "CTC (Connectionist Temporal Classification) decoder-based speech recognition model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

CTC (Connectionist Temporal Classification) decoder-based speech recognition model.

## For Beginners

Traditional speech recognition required aligning audio with text
frame-by-frame, which is expensive and error-prone. CTC removes this requirement:

1. The encoder processes audio features (mel-spectrogram) and outputs a probability

distribution over characters for each time frame.

2. CTC allows the model to output a special "blank" token when it's not sure which

character comes next.

3. The decoder collapses repeated characters and removes blanks to get the final text.

Example: "h h h - e e - l - l - o" becomes "hello" (- = blank, repeated chars collapsed).

Beam search improves accuracy by considering multiple possible decodings simultaneously.

**Usage:**

## How It Works

CTC (Graves et al., 2006) is the most widely-used alignment-free training criterion for
ASR. A CTC decoder pairs a neural encoder (Transformer, Conformer, or LSTM) with a CTC
output head and decodes using greedy or beam search with optional language model rescoring.
This class provides a standalone CTC-based ASR pipeline with configurable beam width and
language model integration.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CTCDecoder(NeuralNetworkArchitecture<>,CTCDecoderOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CTC decoder model in native training mode. |
| `CTCDecoder(NeuralNetworkArchitecture<>,String,CTCDecoderOptions)` | Creates a CTC decoder model in ONNX inference mode. |

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

