---
title: "FullSubNetPlus<T>"
description: "FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) for speech enhancement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Enhancement`

FullSubNet+ (Full-Band and Sub-Band Fusion Network Plus) for speech enhancement.

## For Beginners

FullSubNet+ is a two-part system for cleaning up noisy speech:
one part looks at the big picture (all frequencies), and another focuses on fine details
(small frequency groups). Together they produce cleaner audio than either alone.

**Usage:**

## How It Works

FullSubNet+ (Chen et al., ICASSP 2022) improves upon FullSubNet by using channel-attention-based
full-band models and redesigned sub-band inputs. It achieves PESQ 3.25 and STOI 0.96 on the
DNS Challenge dataset at 16 kHz.

**Architecture:**

- **Full-band model**: LSTM with channel attention across all frequency bins
- **Sub-band model**: LSTM processing local frequency neighborhoods
- **Fusion**: Full-band output guides sub-band processing
- **Complex mask estimation**: Predicts both magnitude and phase corrections

**References:**

- Paper: "FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms" (Chen et al., ICASSP 2022)
- Repository: https://github.com/hit-thusz-RookieCJ/FullSubNet-plus

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FullSubNetPlus(NeuralNetworkArchitecture<>,FullSubNetPlusOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FullSubNet+ model in native training mode. |
| `FullSubNetPlus(NeuralNetworkArchitecture<>,String,FullSubNetPlusOptions)` | Creates a FullSubNet+ model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnhancementStrength` |  |
| `LatencySamples` |  |
| `NumChannels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateAsync(FullSubNetPlusOptions,IProgress<Double>,CancellationToken)` | Downloads and creates a FullSubNet+ model asynchronously. |
| `Enhance(Tensor<>)` |  |
| `EnhanceWithReference(Tensor<>,Tensor<>)` | Enhances audio using a reference noise sample for spectral subtraction before neural enhancement. |
| `EstimateNoiseProfile(Tensor<>)` |  |
| `ProcessChunk(Tensor<>)` |  |
| `ResetState` |  |

