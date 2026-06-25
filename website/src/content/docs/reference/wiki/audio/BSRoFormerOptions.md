---
title: "BSRoFormerOptions"
description: "Configuration options for the BS-RoFormer (Band-Split Rotary Transformer) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the BS-RoFormer (Band-Split Rotary Transformer) model.

## For Beginners

BS-RoFormer separates music into individual instruments by dividing
audio into frequency bands (like bass, mid, treble on an equalizer), processing each band
with AI, then combining the results. The "Rotary" part helps it understand where sounds
occur in time, even for long songs.

## How It Works

BS-RoFormer (Lu et al., 2023) applies band-split processing with rotary position embeddings
to achieve state-of-the-art music source separation. It splits the spectrogram into frequency
bands, processes each with Transformers, and fuses results, achieving 12.8 dB SDR on vocals.

## Properties

| Property | Summary |
|:-----|:--------|
| `BandEmbeddingDim` | Gets or sets the band embedding dimension. |
| `NumBands` | Gets or sets the number of frequency bands to split into. |

