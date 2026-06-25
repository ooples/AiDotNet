---
title: "DACOptions"
description: "Configuration options for the Descript Audio Codec (DAC) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.Effects`

Configuration options for the Descript Audio Codec (DAC) model.

## For Beginners

DAC is like a super-efficient audio compressor. While MP3 typically
uses 128-320 kbps, DAC achieves similar quality at just 8 kbps (16-40x smaller files).
It works by:

1. **Encoding**: Converting audio into compact numerical codes (tokens)
2. **Quantizing**: Discretizing the codes into a small set of entries
3. **Decoding**: Reconstructing audio from the tokens

Unlike EnCodec, DAC uses improved quantization techniques and periodic activations
for better reconstruction of music and complex audio.

## How It Works

DAC (Kumar et al., 2024, Descript) is a high-fidelity universal neural audio codec
that compresses audio to 8 kbps while maintaining near-lossless quality. It uses
residual vector quantization (RVQ) with improved codebook usage, periodic activation
functions (Snake), and multi-scale STFT discriminators. DAC handles speech, music,
and environmental sounds at 16/24/44.1 kHz.

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookDim` | Gets or sets the codebook dimension. |
| `CodebookSize` | Gets or sets the codebook size (entries per codebook). |
| `CommitmentLossWeight` | Gets or sets the commitment loss weight for codebook training. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `EncoderChannels` | Gets or sets the number of encoder channels per downsampling stage. |
| `EncoderDim` | Gets or sets the encoder hidden dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumChannels` | Gets or sets the number of audio channels. |
| `NumCodebooks` | Gets or sets the number of codebooks (residual quantization levels). |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `SampleRate` | Gets or sets the audio sample rate in Hz. |
| `TargetBitrate` | Gets or sets the target bitrate in kbps. |
| `TokenFrameRate` | Gets or sets the frame rate of encoded tokens (tokens per second). |
| `Variant` | Gets or sets the model variant ("16khz", "24khz", "44khz"). |

