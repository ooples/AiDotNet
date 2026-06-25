---
title: "NeuralParametricEQ<T>"
description: "Neural Parametric EQ model for automatic equalization (Steinmetz et al., 2022)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Neural Parametric EQ model for automatic equalization (Steinmetz et al., 2022).

## For Beginners

Imagine you have audio that sounds too bassy or too bright.
Instead of manually turning EQ knobs, this model listens to the audio and automatically
figures out the right EQ settings. It predicts gain (louder/quieter), frequency (which
range to adjust), and Q (how narrow the adjustment is) for each band.

**Usage:**

## How It Works

Neural Parametric EQ uses a neural network to automatically predict optimal parametric
EQ settings (gain, frequency, Q for each band) to match a target frequency response.
It analyzes input audio and outputs the parameters for a cascaded biquad filter bank,
enabling automatic mastering, hearing aid fitting, and frequency matching.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralParametricEQ(NeuralNetworkArchitecture<>,NeuralParametricEQOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a Neural Parametric EQ model in native training mode. |
| `NeuralParametricEQ(NeuralNetworkArchitecture<>,String,NeuralParametricEQOptions)` | Creates a Neural Parametric EQ model in ONNX inference mode. |

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
| `EnhanceWithReference(Tensor<>,Tensor<>)` | Enhances audio by applying parametric EQ. |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |

