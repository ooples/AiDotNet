---
title: "SpikingFullSubNetOptions"
description: "Configuration options for the Spiking-FullSubNet model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Enhancement`

Configuration options for the Spiking-FullSubNet model.

## For Beginners

This model works like the brain's neurons - they "fire" or "don't fire"
(binary spikes) instead of using continuous values. This makes it much more energy-efficient
while still cleaning up noisy audio effectively. Great for battery-powered devices.

## How It Works

Spiking-FullSubNet (Yu et al., 2023) replaces traditional activation functions in the
FullSubNet architecture with spiking neural network (SNN) neurons, achieving comparable
speech enhancement quality with significantly lower computational cost (energy efficiency).
It combines full-band and sub-band processing with bio-inspired spiking activations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EnhancementStrength` | Gets or sets the enhancement strength (0.0-1.0). |
| `FftSize` | Gets or sets the FFT size. |
| `FullBandHiddenSize` | Gets or sets the full-band hidden size. |
| `HopLength` | Gets or sets the hop length. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFreqBins` | Gets or sets the number of frequency bins. |
| `NumFullBandLayers` | Gets or sets the number of full-band layers. |
| `NumSubBandLayers` | Gets or sets the number of sub-band layers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `SpikingThreshold` | Gets or sets the spiking neuron threshold. |
| `SubBandHiddenSize` | Gets or sets the sub-band hidden size. |
| `TimeConstant` | Gets or sets the spiking neuron time constant. |
| `Variant` | Gets or sets the model variant. |

