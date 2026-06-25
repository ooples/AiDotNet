---
title: "FDYSED<T>"
description: "FDY-SED (Frequency Dynamic Sound Event Detection) model for DCASE-winning SED."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Classification`

FDY-SED (Frequency Dynamic Sound Event Detection) model for DCASE-winning SED.

## For Beginners

FDY-SED is like having a team of audio specialists:

- A bass expert identifies low rumbles (washing machines, traffic, thunder)
- A midrange expert identifies speech, barks, and music
- A treble expert identifies high-pitched sounds (birds, alarms, glass breaking)

Each expert has their own set of filters optimized for their frequency range, then they
combine their findings to produce the final detection.

**Usage:**

## How It Works

FDY-SED (Nam et al., ICASSP 2022) introduces frequency-dynamic convolutions for sound event
detection, where convolution kernels are dynamically generated based on the frequency band
being processed. Combined with mean-teacher semi-supervised training, it achieves DCASE
challenge-winning results on the DESED (Domestic Environment Sound Event Detection) dataset.

**Architecture:**

- **Frequency-dynamic CNN**: 5 CNN blocks with frequency-dynamic convolutions. Each block

generates different kernels for different frequency groups, allowing specialized processing
for bass, mid, and treble ranges.

- **Bidirectional GRU**: 2 layers of bidirectional GRU for temporal modeling across frames.
- **Frame-level classifier**: Sigmoid-activated linear layer for multi-label frame-level detection.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FDYSED(NeuralNetworkArchitecture<>,FDYSEDOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an FDY-SED model for native training mode. |
| `FDYSED(NeuralNetworkArchitecture<>,String,FDYSEDOptions)` | Creates an FDY-SED model for ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EventLabels` |  |
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

## Fields

| Field | Summary |
|:-----|:--------|
| `AudioSetLabels` | AudioSet-527 standard event labels. |

