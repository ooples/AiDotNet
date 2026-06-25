---
title: "SpikingFullSubNet<T>"
description: "Spiking-FullSubNet speech enhancement model using spiking neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

Spiking-FullSubNet speech enhancement model using spiking neural networks.

## For Beginners

Spiking-FullSubNet cleans up noisy audio using a brain-inspired approach.
Instead of traditional neural networks that pass numbers between layers, it uses "spikes"
(on/off signals) like real neurons. This means it can run much more efficiently on
specialized hardware while still producing clean, clear speech.

**Usage:**

## How It Works

Spiking-FullSubNet (2024) replaces conventional RNNs with spiking neural network (SNN)
layers in the FullSubNet architecture. SNNs use biologically-inspired binary spike events
instead of continuous activations, achieving comparable speech enhancement quality with
significantly reduced energy consumption, making it suitable for neuromorphic hardware.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpikingFullSubNet(NeuralNetworkArchitecture<>,SpikingFullSubNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Spiking-FullSubNet model in native training mode. |
| `SpikingFullSubNet(NeuralNetworkArchitecture<>,String,SpikingFullSubNetOptions)` | Creates a Spiking-FullSubNet model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

