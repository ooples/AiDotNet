---
title: "AudioLDMClassifierOptions"
description: "Configuration options for the AudioLDM Classifier model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Classification`

Configuration options for the AudioLDM Classifier model.

## For Beginners

AudioLDM was originally built to generate audio from text, but
it turns out the internal features it learned are also great for recognizing audio.
This classifier uses those learned features to identify sounds, similar to how image
generation models can also be used for image recognition.

## How It Works

AudioLDM Classifier (Liu et al., 2023) repurposes the latent representations from the
AudioLDM diffusion model for audio classification. By extracting intermediate features
from the AudioLDM U-Net, it achieves strong classification performance by leveraging
the rich audio representations learned during generative pre-training.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierDim` | Gets or sets the classifier hidden dimension. |
| `CustomLabels` | Gets or sets custom event labels. |
| `DetectionWindowSize` | Gets or sets the window size in seconds for detection. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `FMax` | Gets or sets the maximum frequency for mel filterbank. |
| `FMin` | Gets or sets the minimum frequency for mel filterbank. |
| `FftSize` | Gets or sets the FFT window size. |
| `HopLength` | Gets or sets the hop length. |
| `LatentDim` | Gets or sets the latent dimension from AudioLDM. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumClassifierLayers` | Gets or sets the number of classifier layers. |
| `NumMels` | Gets or sets the number of mel filterbank channels. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `Threshold` | Gets or sets the confidence threshold for event detection. |
| `Variant` | Gets or sets the model variant. |
| `WindowOverlap` | Gets or sets the window overlap ratio (0-1). |

