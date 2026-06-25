---
title: "ParametricEqualizer<T>"
description: "Multi-band parametric equalizer effect."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Multi-band parametric equalizer effect.

## For Beginners

An equalizer adjusts the volume of different frequencies!

Think of it like a graphic equalizer on a stereo:

- Bass slider controls low frequencies (20-200 Hz)
- Mid slider controls middle frequencies (200-4000 Hz)
- Treble slider controls high frequencies (4000-20000 Hz)

A PARAMETRIC EQ is more flexible:

- Frequency: Which frequency to adjust (e.g., 1000 Hz)
- Gain: How much to boost/cut (e.g., +6 dB)
- Q (bandwidth): How wide the adjustment is (narrow = surgical, wide = gentle)

Filter types:

- Low Shelf: Boosts/cuts all frequencies below a point
- High Shelf: Boosts/cuts all frequencies above a point
- Peak (Bell): Boosts/cuts around a specific frequency
- Low Pass: Removes frequencies above a point
- High Pass: Removes frequencies below a point

Common EQ moves:

- Cut 200-400 Hz: Reduce muddiness
- Boost 3-5 kHz: Add presence to vocals
- Cut 2-4 kHz: Reduce harshness
- Boost 10+ kHz: Add "air" and sparkle
- High pass at 80 Hz: Remove rumble from vocals

## How It Works

A parametric EQ allows precise control over frequency response with
adjustable frequency, gain, and bandwidth (Q) for each band.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ParametricEqualizer(Int32,Double)` | Creates a parametric EQ with a default 5-band configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Bands` | Gets the EQ bands. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBand(Double,Double,Double,EqFilterType)` | Adds an EQ band. |
| `ProcessSampleInternal()` |  |
| `RemoveBand(Int32)` | Removes an EQ band by index. |
| `Reset` |  |
| `SetBand(Int32,Double,Double,Double)` | Sets band parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bands` | EQ bands. |

