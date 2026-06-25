---
title: "Compressor<T>"
description: "Dynamic range compressor effect."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Dynamic range compressor effect.

## For Beginners

A compressor is like an automatic volume control!

The problem it solves:

- Singer gets too close to mic → TOO LOUD
- Singer backs away → too quiet
- Drums hit hard → peaks clip

What a compressor does:

- Watches the signal level
- When it exceeds the threshold, it turns down the volume
- The ratio controls how much it turns down (4:1 means 4dB over becomes 1dB over)

Key parameters:

- Threshold: Level above which compression starts (dB)
- Ratio: How aggressively to compress (2:1 = gentle, 20:1 = limiting)
- Attack: How quickly compression kicks in (ms)
- Release: How quickly compression releases (ms)
- Makeup Gain: Boost to compensate for reduced volume (dB)

Common uses:

- Vocals: Even out dynamics (ratio 3:1 to 6:1)
- Drums: Punch and sustain (ratio 4:1 to 8:1)
- Bass: Consistent level (ratio 4:1 to 6:1)
- Master bus: Glue the mix together (ratio 2:1 to 4:1)
- Podcasts: Keep voice at consistent level

## How It Works

A compressor reduces the dynamic range of audio by attenuating signals
that exceed a threshold. This makes quiet parts louder relative to loud parts.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Compressor(Int32,Double,Double,Double,Double,Double,Double,Double)` | Creates a compressor with default broadcast settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetGainReduction` | Gets the current gain reduction in dB. |
| `OnParameterChanged(String,)` |  |
| `ProcessSampleInternal()` |  |
| `Reset` |  |

