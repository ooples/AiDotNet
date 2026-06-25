---
title: "MelBandRoFormerOptions"
description: "Configuration options for the MelBand-RoFormer model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Audio.SourceSeparation`

Configuration options for the MelBand-RoFormer model.

## For Beginners

This is an improved version of BS-RoFormer that divides frequencies
using the mel scale (which matches how humans hear) instead of equal-width bands.
This means more detail in the frequencies that matter most to us.

## How It Works

MelBand-RoFormer (2024) extends BS-RoFormer by using mel-scale frequency bands instead of
linear bands, better matching human perception. It achieves state-of-the-art SDR on vocals
(13.2 dB) and other stems on the MUSDB18-HQ benchmark.

