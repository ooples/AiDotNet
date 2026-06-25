---
title: "SoundStreamOptions"
description: "Configuration options for the SoundStream neural audio codec model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the SoundStream neural audio codec model.

## For Beginners

SoundStream is Google's version of a neural audio compressor.
It squeezes audio into tiny tokens and reconstructs it back. The key innovation is
"residual vector quantization" (RVQ) - it uses multiple codebooks, each one refining
the previous approximation, like painting with increasingly fine brushstrokes.

## How It Works

SoundStream (Zeghidour et al., 2021, Google) is a neural audio codec that compresses
audio at 3-18 kbps using a fully convolutional encoder-decoder with residual vector
quantization. It pioneered the RVQ approach later adopted by EnCodec, and powers
Google's AudioLM and MusicLM systems.

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets or sets the number of audio channels. |
| `CodebookDim` | Gets or sets the codebook dimension. |
| `CodebookSize` | Gets or sets the codebook size per quantizer. |
| `DownsampleRatios` | Gets or sets the temporal downsampling ratios. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the encoder channel dimensions. |
| `EncoderDim` | Gets or sets the encoder output dimension (embedding dimension). |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumQuantizers` | Gets or sets the number of residual vector quantizers. |
| `NumResBlocks` | Gets or sets the number of residual blocks per stage. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate. |
| `TargetBitrateKbps` | Gets or sets the target bitrate in kbps. |

