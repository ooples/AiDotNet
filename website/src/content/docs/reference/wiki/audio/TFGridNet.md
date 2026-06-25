---
title: "TFGridNet<T>"
description: "TF-GridNet (Time-Frequency GridNet) for speech enhancement and separation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

TF-GridNet (Time-Frequency GridNet) for speech enhancement and separation.

## For Beginners

TF-GridNet cleans audio by processing a time-frequency grid in two
alternating directions: across frequencies (understanding harmonic structure) and across
time (tracking how sounds evolve). This grid approach captures both local and global patterns.

**Usage:**

## How It Works

TF-GridNet (Wang et al., ICASSP 2023) applies alternating attention along the time and frequency
axes in a grid pattern, achieving 23.4 dB SI-SNRi on WSJ0-2mix and PESQ 3.41 on DNS Challenge.

**Architecture:**

- **Input embedding**: Maps complex STFT to per-bin embeddings
- **Grid blocks**: Each block has intra-frame (frequency) and inter-frame (time) modules
- **Intra-frame**: LSTM/attention across frequency bins for each time frame
- **Inter-frame**: LSTM/attention across time frames for each frequency bin
- **Output**: Reconstructs complex STFT for synthesis

**References:**

- Paper: "TF-GridNet: Making Time-Frequency Domain Models Great Again" (Wang et al., ICASSP 2023)
- Repository: https://github.com/espnet/espnet

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TFGridNet(NeuralNetworkArchitecture<>,String,TFGridNetOptions)` | Creates a TF-GridNet model in ONNX inference mode. |
| `TFGridNet(NeuralNetworkArchitecture<>,TFGridNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TF-GridNet model in native training mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(TFGridNetOptions,IProgress<Double>,CancellationToken)` | Downloads and creates a TF-GridNet model asynchronously. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` |  |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

