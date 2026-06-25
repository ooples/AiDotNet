---
title: "DAC<T>"
description: "Descript Audio Codec (DAC) - high-fidelity universal neural audio codec (Kumar et al., 2024, Descript)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Descript Audio Codec (DAC) - high-fidelity universal neural audio codec (Kumar et al., 2024, Descript).

## For Beginners

DAC is like a super-efficient audio compressor. While MP3 typically
uses 128-320 kbps, DAC achieves similar quality at just 8 kbps (16-40x smaller files).
It works by:

1. **Encoding**: Converting audio into compact numerical codes (tokens)
2. **Quantizing**: Discretizing the codes using improved residual vector quantization
3. **Decoding**: Reconstructing audio from the tokens using Snake activations

Key improvements over EnCodec:

- Better codebook utilization (more of the codebook entries are actually used)
- Snake activations for better periodic signal reconstruction (important for music)
- Works with any audio type, not just speech

**Usage:**

## How It Works

DAC is a high-fidelity neural audio codec that compresses audio to approximately 8 kbps
while maintaining near-lossless quality. It uses residual vector quantization (RVQ) with
improved codebook utilization, periodic activation functions (Snake activations), and
multi-scale STFT discriminators. Unlike EnCodec which was designed primarily for speech,
DAC is universal - handling speech, music, and environmental sounds at 16/24/44.1 kHz.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DAC(NeuralNetworkArchitecture<>,DACOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DAC model in native training mode. |
| `DAC(NeuralNetworkArchitecture<>,String,DACOptions)` | Creates a DAC model in ONNX inference mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` |  |
| `NumQuantizers` |  |
| `TokenFrameRate` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Int32[0:,0:])` |  |
| `DecodeAsync(Int32[0:,0:],CancellationToken)` |  |
| `DecodeEmbeddings(Tensor<>)` |  |
| `Encode(Tensor<>)` |  |
| `EncodeAsync(Tensor<>,CancellationToken)` |  |
| `EncodeEmbeddings(Tensor<>)` |  |
| `GetBitrate(Nullable<Int32>)` |  |

