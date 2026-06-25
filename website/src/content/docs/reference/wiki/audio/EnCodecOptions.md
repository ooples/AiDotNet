---
title: "EnCodecOptions"
description: "Configuration options for the EnCodec neural audio codec model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Generation`

Configuration options for the EnCodec neural audio codec model.

## For Beginners

EnCodec is like an AI-powered MP3. It compresses audio into tiny
tokens (numbers) and decompresses them back. At 6 kbps it sounds almost as good as the
original, while MP3 needs 128 kbps for similar quality. The compressed tokens are also
used by AI models that generate speech and music.

## How It Works

EnCodec (Defossez et al., 2022, Meta) is a neural audio codec that compresses audio to
1.5-24 kbps using residual vector quantization (RVQ). It uses an encoder-decoder architecture
with a multi-scale discriminator for adversarial training, achieving near-transparent quality
at 6 kbps. EnCodec tokens are widely used as input for audio language models.

## Properties

| Property | Summary |
|:-----|:--------|
| `Channels` | Gets or sets the number of audio channels (1=mono, 2=stereo). |
| `CodebookDim` | Gets or sets the codebook dimension. |
| `CodebookSize` | Gets or sets the codebook size per quantizer. |
| `DecoderModelPath` | Gets or sets the path to the ONNX decoder model. |
| `DownsampleRatios` | Gets or sets the downsampling ratios at each stage. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the encoder channel dimensions at each downsampling stage. |
| `EncoderDim` | Gets or sets the encoder output dimension. |
| `EncoderModelPath` | Gets or sets the path to the ONNX encoder model. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the combined ONNX model path. |
| `NumQuantizers` | Gets or sets the number of residual vector quantizers. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate (24 kHz or 48 kHz). |
| `TargetBandwidthKbps` | Gets or sets the target bandwidth in kbps (determines number of active quantizers). |

