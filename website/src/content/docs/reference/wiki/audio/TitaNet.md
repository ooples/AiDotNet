---
title: "TitaNet<T>"
description: "TitaNet speaker verification and embedding extraction model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

TitaNet speaker verification and embedding extraction model.

## For Beginners

TitaNet is NVIDIA's advanced voice fingerprinting model. It listens
to someone speaking and creates a compact vector that uniquely identifies their voice.
You can use it to verify a speaker's identity or find who's speaking among known voices.
It comes in three sizes: Small (6M params), Medium (13M params), and Large (25M params).

**Usage:**

## How It Works

TitaNet (Koluguri et al., ICASSP 2022) is NVIDIA's speaker embedding model based on
1D depth-wise separable convolutions with Squeeze-Excitation and global context.
TitaNet-L achieves 0.68% EER on VoxCeleb1-O, outperforming ECAPA-TDNN.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TitaNet(NeuralNetworkArchitecture<>,String,TitaNetOptions)` | Creates a TitaNet speaker model in ONNX inference mode. |
| `TitaNet(NeuralNetworkArchitecture<>,TitaNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TitaNet speaker model in native training mode. |

