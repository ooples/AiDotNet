---
title: "MultiBandMelGAN<T>"
description: "Multi-band MelGAN: decomposes target into sub-bands, generates each in parallel, then synthesizes full-band."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TextToSpeech.Vocoders`

Multi-band MelGAN: decomposes target into sub-bands, generates each in parallel, then synthesizes full-band.

## For Beginners

Multi-band MelGAN: decomposes target into sub-bands, generates each in parallel, then synthesizes full-band.. This model converts text input into speech audio output.

## How It Works

**References:**

- Paper: "Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech" (Yang et al., 2021)

## Methods

| Method | Summary |
|:-----|:--------|
| `MelToWaveform(Tensor<>)` | Converts mel to waveform using Multi-band MelGAN's sub-band parallel generation. |

