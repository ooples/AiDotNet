---
title: "CREPEOptions"
description: "Configuration options for the CREPE (Convolutional Representation for Pitch Estimation) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.MusicAnalysis`

Configuration options for the CREPE (Convolutional Representation for Pitch Estimation) model.

## For Beginners

CREPE detects the pitch (how high or low a note sounds) from audio.
It works by analyzing small windows of audio through a neural network that outputs which
frequency (pitch) is most likely. The model uses 360 pitch bins spanning from C1 (32.7 Hz)
to B7 (1975.5 Hz) with 20-cent resolution.

## How It Works

CREPE (Kim et al., 2018) is a deep learning model for monophonic pitch detection. It uses
a convolutional architecture trained on synthesized audio to predict pitch with high accuracy,
outperforming traditional methods (YIN, pYIN) especially on noisy or challenging audio.

## Properties

| Property | Summary |
|:-----|:--------|
| `CapacityMultiplier` | Gets or sets the capacity multiplier for convolutional filter counts. |
| `CentsPerBin` | Gets or sets the cents per bin resolution. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FrameSize` | Gets or sets the input frame size in samples (CREPE uses 1024 samples at 16 kHz). |
| `HopLength` | Gets or sets the hop length between frames in samples. |
| `LearningRate` | Gets or sets the learning rate for training. |
| `MaxFrequency` | Gets or sets the maximum frequency in Hz (B7). |
| `MinFrequency` | Gets or sets the minimum frequency in Hz (C1). |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumBins` | Gets or sets the number of pitch bins (20-cent resolution from C1 to B7). |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the expected audio sample rate in Hz. |
| `UseViterbiDecoding` | Gets or sets whether to use Viterbi decoding for pitch smoothing. |
| `Variant` | Gets or sets the model capacity variant. |
| `VoicingThreshold` | Gets or sets the voicing confidence threshold. |

