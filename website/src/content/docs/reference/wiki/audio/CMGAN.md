---
title: "CMGAN<T>"
description: "CMGAN (Conformer-based Metric GAN) for speech enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

CMGAN (Conformer-based Metric GAN) for speech enhancement.

## For Beginners

CMGAN uses a competition between two networks: a "generator" that
cleans audio and a "discriminator" that judges quality. The generator uses Conformer
layers that combine attention (understanding context) with convolution (local patterns).

**Usage:**

## How It Works

CMGAN (Cao et al., INTERSPEECH 2022) combines a conformer-based generator with a metric
discriminator for high-quality speech enhancement, achieving PESQ 3.41 and STOI 0.97
on the VoiceBank-DEMAND dataset.

**Architecture:**

- **U-Net encoder**: Compresses the noisy spectrogram with convolutional blocks
- **Conformer bottleneck**: Self-attention + convolution for global context
- **U-Net decoder**: Reconstructs clean spectrogram with skip connections
- **Metric discriminator**: Judges enhancement quality during training

**References:**

- Paper: "CMGAN: Conformer-based Metric GAN for Speech Enhancement" (Cao et al., INTERSPEECH 2022)
- Repository: https://github.com/ruizhecao96/CMGAN

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CMGAN(NeuralNetworkArchitecture<>,CMGANOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CMGAN model in native training mode. |
| `CMGAN(NeuralNetworkArchitecture<>,String,CMGANOptions)` | Creates a CMGAN model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(CMGANOptions,IProgress<Double>,CancellationToken)` | Downloads and creates a CMGAN model asynchronously. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

