---
title: "HTDemucsOptions"
description: "Configuration options for the HTDemucs (Hybrid Transformer Demucs) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the HTDemucs (Hybrid Transformer Demucs) model.

## For Beginners

HTDemucs is Meta's best music separator. It works in two ways at once:
one path processes the raw audio waves, another processes the frequency picture (spectrogram).
A Transformer helps both paths share information, giving the best of both worlds.

## How It Works

HTDemucs (Rouard et al., ICASSP 2023) is Meta's hybrid architecture combining a temporal
convolutional encoder with cross-domain Transformer attention. It operates on both waveform
and spectrogram simultaneously, achieving 9.0 dB SDR on MUSDB18-HQ.

## Properties

| Property | Summary |
|:-----|:--------|
| `EncoderChannels` | Gets or sets the encoder channel progression. |
| `TemporalKernelSize` | Gets or sets the temporal kernel size. |
| `TemporalStride` | Gets or sets the temporal stride. |

