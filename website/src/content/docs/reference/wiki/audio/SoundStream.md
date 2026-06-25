---
title: "SoundStream<T>"
description: "SoundStream neural audio codec from Google for efficient audio compression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Generation`

SoundStream neural audio codec from Google for efficient audio compression.

## For Beginners

SoundStream compresses audio using AI, like a smart zip file for
sound. It can compress a song to just 3-6 kbps (versus 128 kbps for MP3) while keeping
good quality. It uses "residual vector quantization" which is like describing a painting
with increasingly fine details: the first pass captures the rough shape, each additional
pass adds more nuance.

**Usage:**

## How It Works

SoundStream (Zeghidour et al., 2021, Google) is a fully convolutional encoder-decoder
with residual vector quantization for audio compression at 3-18 kbps. It pioneered the
RVQ approach later adopted by EnCodec. SoundStream powers Google's AudioLM and MusicLM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SoundStream(NeuralNetworkArchitecture<>,SoundStreamOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SoundStream model in native training mode. |
| `SoundStream(NeuralNetworkArchitecture<>,String,SoundStreamOptions)` | Creates a SoundStream model in ONNX inference mode. |

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

