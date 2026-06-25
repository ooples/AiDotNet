---
title: "FDYSEDOptions"
description: "Configuration options for the FDY-SED (Frequency Dynamic Sound Event Detection) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the FDY-SED (Frequency Dynamic Sound Event Detection) model.

## For Beginners

FDY-SED is like having a team of specialized listeners, where each
listener is an expert at hearing different pitch ranges:

- One listener is good at hearing low rumbles (bass) like washing machines or traffic
- Another is good at mid-range sounds like speech or dog barks
- Another specializes in high-pitched sounds like bird chirps or alarms

Instead of using the same "ear" for all frequencies (like a standard CNN), FDY-SED adapts
its filters based on which frequency range it's analyzing. This makes it more accurate at
detecting sounds that have very different spectral characteristics.

## How It Works

FDY-SED (Nam et al., ICASSP 2022) introduces frequency-dynamic convolutions for sound event
detection, where convolution kernels are dynamically generated based on the frequency band being
processed. This allows the model to apply different processing strategies to different frequency
ranges (e.g., bass frequencies need different patterns than treble). FDY-SED achieves DCASE
challenge-winning results on the DESED dataset for domestic sound event detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `CNNChannels` | Gets or sets the number of CNN channels per block. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DetectionWindowSize` | Gets or sets the window size in seconds for detection. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EmbeddingDim` | Gets or sets the embedding dimension after CNN blocks. |
| `FDYKernelHiddenSize` | Gets or sets the frequency dynamic kernel generation hidden size. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the learning rate. |
| `MeanTeacherWeight` | Gets or sets the mean teacher consistency weight for semi-supervised training. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumFrequencyGroups` | Gets or sets the number of frequency dynamic convolution groups. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumRNNLayers` | Gets or sets the number of RNN layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `RNNHiddenSize` | Gets or sets the RNN hidden size for temporal modeling. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `Variant` | Gets or sets the model variant ("small", "base", "large"). |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1). |

