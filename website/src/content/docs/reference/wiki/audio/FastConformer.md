---
title: "FastConformer<T>"
description: "Fast Conformer speech recognition model (Rekesh et al., 2023, NVIDIA NeMo)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SpeechRecognition`

Fast Conformer speech recognition model (Rekesh et al., 2023, NVIDIA NeMo).

## For Beginners

Fast Conformer is NVIDIA's speed-optimized version of the Conformer.
It compresses audio early on (8x downsampling) so the expensive transformer layers process
much shorter sequences. Think of it as reading a summary instead of the full book - same
information, much faster processing.

Key advantages:

- 2.4x faster than standard Conformer
- Same or better accuracy
- Great for long audio (podcasts, meetings, lectures)
- Supports both CTC and RNN-T decoding

**Usage:**

## How It Works

Fast Conformer (Rekesh et al., 2023, NVIDIA NeMo) is an optimized Conformer variant
with 8x depthwise-separable convolution downsampling in the front-end, reducing the
sequence length early and enabling efficient processing of long audio. Combined with
multi-blank CTC or RNN-T, it achieves 2.4x speedup over standard Conformer with no
accuracy loss. It reaches WER 1.8%/3.4% on LibriSpeech test-clean/other.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FastConformer(NeuralNetworkArchitecture<>,FastConformerOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Fast Conformer model in native training mode. |
| `FastConformer(NeuralNetworkArchitecture<>,String,FastConformerOptions)` | Creates a Fast Conformer model in ONNX inference mode. |

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

