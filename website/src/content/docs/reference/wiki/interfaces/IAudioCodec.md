---
title: "IAudioCodec<T>"
description: "Defines the contract for neural audio codecs that compress and decompress audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for neural audio codecs that compress and decompress audio.

## For Beginners

A neural audio codec is like an AI-powered audio compressor.
It converts audio into a very compact code (tokens), then converts that code back to audio.
Think of it like a zip file for audio, but using AI instead of traditional compression.

Two key operations:

- Encode: Audio waveform -> compact tokens (small numbers)
- Decode: Compact tokens -> reconstructed audio waveform

Uses:

- Ultra-low bitrate audio streaming (1-6 kbps vs 128 kbps for MP3)
- Audio tokens for AI language models
- High-quality audio compression for storage

## How It Works

Neural audio codecs use neural networks to compress audio into compact discrete tokens
and reconstruct audio from those tokens. They achieve much higher compression ratios
than traditional codecs (MP3, AAC) at comparable quality. The tokens can also serve
as input to language models for audio generation.

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookSize` | Gets the codebook size (vocabulary size per quantizer). |
| `NumQuantizers` | Gets the number of quantizer levels (codebooks). |
| `SampleRate` | Gets the sample rate of audio this codec operates on. |
| `TokenFrameRate` | Gets the frame rate of the encoded tokens (tokens per second). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Decode(Int32[0:,0:])` | Decodes tokens back into audio. |
| `DecodeAsync(Int32[0:,0:],CancellationToken)` | Decodes tokens asynchronously. |
| `DecodeEmbeddings(Tensor<>)` | Reconstructs audio from continuous embeddings. |
| `Encode(Tensor<>)` | Encodes audio into discrete tokens. |
| `EncodeAsync(Tensor<>,CancellationToken)` | Encodes audio asynchronously. |
| `EncodeEmbeddings(Tensor<>)` | Encodes audio into continuous embeddings (before quantization). |
| `GetBitrate(Nullable<Int32>)` | Gets the bitrate at the given number of quantizers in bits per second. |

