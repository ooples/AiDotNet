---
title: "AudioLDMClassifier<T>"
description: "AudioLDM Classifier that repurposes AudioLDM's latent representations for audio event detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

AudioLDM Classifier that repurposes AudioLDM's latent representations for audio event detection.

## For Beginners

AudioLDM was originally built to create audio from text descriptions.
But it turns out the "understanding" it developed during training is also great for
recognizing sounds. This classifier reuses that understanding: instead of generating audio,
it identifies what sounds are present in a recording.

**Usage:**

## How It Works

AudioLDM Classifier (Liu et al., 2023) extracts intermediate features from the AudioLDM
diffusion U-Net to build a strong audio classifier. Since AudioLDM was trained to generate
audio from text, its internal representations capture rich, semantically meaningful audio
features that transfer well to classification tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioLDMClassifier(NeuralNetworkArchitecture<>,AudioLDMClassifierOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an AudioLDM Classifier in native training mode. |
| `AudioLDMClassifier(NeuralNetworkArchitecture<>,String,AudioLDMClassifierOptions)` | Creates an AudioLDM Classifier in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsOnnxMode` |  |
| `StandardLabels` | Standard AudioSet labels for audio event detection. |
| `SupportedEvents` |  |
| `TimeResolution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` |  |
| `Detect(Tensor<>,)` |  |
| `DetectAsync(Tensor<>,CancellationToken)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>)` |  |
| `DetectSpecific(Tensor<>,IReadOnlyList<String>,)` |  |
| `GetEventProbabilities(Tensor<>)` |  |
| `StartStreamingSession` |  |
| `StartStreamingSession(Int32,)` |  |

