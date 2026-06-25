---
title: "MatchaTTS<T>"
description: "Matcha-TTS fast text-to-speech model using conditional flow matching (Mehta et al., 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.TextToSpeech`

Matcha-TTS fast text-to-speech model using conditional flow matching (Mehta et al., 2024).

## For Beginners

Matcha-TTS is a fast, lightweight text-to-speech model.
While other models need many steps to gradually refine audio (like slowly developing
a photograph), Matcha-TTS takes a shortcut - it finds the most direct path from
random noise to a mel-spectrogram in just a few steps.

Pipeline:

1. Text encoder: Analyzes the input text
2. Duration predictor: Decides how long each sound should be
3. Flow matching decoder: Generates the mel-spectrogram in 2-4 steps
4. Vocoder (separate): Converts mel-spectrogram to audio waveform

**Usage:**

## How It Works

Matcha-TTS uses Optimal Transport Conditional Flow Matching (OT-CFM) to generate
mel-spectrograms from text in just 2-4 synthesis steps, achieving 10x speedup over
Grad-TTS with comparable quality (MOS 4.04 on LJSpeech). The model combines a
text encoder with a U-Net-based flow matching decoder and a duration predictor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatchaTTS(NeuralNetworkArchitecture<>,MatchaTTSOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Matcha-TTS model in native training mode. |
| `MatchaTTS(NeuralNetworkArchitecture<>,String,MatchaTTSOptions)` | Creates a Matcha-TTS model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableVoices` |  |
| `SupportsEmotionControl` |  |
| `SupportsStreaming` |  |
| `SupportsVoiceCloning` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractSpeakerEmbedding(Tensor<>)` |  |
| `StartStreamingSession(String,Double)` |  |
| `Synthesize(String,String,Double,Double)` |  |
| `SynthesizeAsync(String,String,Double,Double,CancellationToken)` |  |
| `SynthesizeWithEmotion(String,String,Double,String,Double)` |  |
| `SynthesizeWithVoiceCloning(String,Tensor<>,Double,Double)` |  |

