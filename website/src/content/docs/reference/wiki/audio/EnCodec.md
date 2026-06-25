---
title: "EnCodec<T>"
description: "EnCodec neural audio codec from Meta for high-fidelity audio compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

EnCodec neural audio codec from Meta for high-fidelity audio compression.

## For Beginners

EnCodec is like a super-efficient audio compressor. Regular MP3 needs
128 kbps for good quality; EnCodec achieves similar quality at just 6 kbps. It works by:

1. Encoding audio into a compact representation
2. Quantizing that into discrete tokens (like words)
3. Decoding those tokens back into audio

**Usage:**

## How It Works

EnCodec (Defossez et al., 2022) compresses audio to 1.5-24 kbps using an encoder-decoder
with residual vector quantization (RVQ). At 6 kbps it achieves near-transparent quality.
EnCodec tokens serve as the audio representation for language models like MusicGen and VALL-E.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnCodec(NeuralNetworkArchitecture<>,EnCodecOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an EnCodec model in native training mode. |
| `EnCodec(NeuralNetworkArchitecture<>,String,EnCodecOptions)` | Creates an EnCodec model in ONNX inference mode. |

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

