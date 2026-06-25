---
title: "CRNNEventDetectorOptions"
description: "Configuration options for the CRNN (Convolutional Recurrent Neural Network) Sound Event Detector."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the CRNN (Convolutional Recurrent Neural Network) Sound Event Detector.

## For Beginners

CRNN is a classic and reliable approach to detecting sounds over time.
Think of it in two stages:

1. **CNN stage**: Looks at the spectrogram like a picture, finding patterns like "this

frequency pattern looks like a dog bark" or "this shape looks like a piano note"

2. **RNN stage**: Reads the patterns over time like reading a sentence, understanding

"the dog bark started, continued, and then stopped"

This combination makes CRNN good at both identifying what sounds are present AND when they
happen. It's simpler than Transformer-based models but still very effective.

## How It Works

CRNN for SED (Cakir et al., 2017) combines convolutional layers for spectral feature extraction
with recurrent layers (GRU/LSTM) for temporal modeling. It is the standard baseline architecture
for the DCASE Sound Event Detection challenge and achieves strong results on AudioSet, ESC-50,
and URBAN-SED benchmarks. The model processes mel spectrograms through CNN blocks, then models
temporal dependencies with bidirectional GRU layers, producing frame-level event probabilities.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bidirectional` | Gets or sets whether to use bidirectional RNN. |
| `CNNChannels` | Gets or sets the number of CNN channels per block. |
| `CNNKernelSize` | Gets or sets the CNN kernel size. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DetectionWindowSize` | Gets or sets the window size in seconds for detection. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length between FFT frames. |
| `LabelSmoothing` | Gets or sets the label smoothing factor. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to a pre-trained ONNX model file. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `NumRNNLayers` | Gets or sets the number of RNN layers. |
| `OnnxOptions` | Gets or sets ONNX runtime options. |
| `PoolSize` | Gets or sets the CNN pooling size for frequency dimension. |
| `RNNHiddenSize` | Gets or sets the RNN hidden size. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `Variant` | Gets or sets the model variant ("small", "base", "large"). |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1). |

