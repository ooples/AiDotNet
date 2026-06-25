---
title: "IAudioEffect<T>"
description: "Defines the contract for audio effects processors."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for audio effects processors.

## For Beginners

Audio effects are like Instagram filters for sound!

Common effects explained:

- Reverb: Adds room ambience (makes it sound like you're in a hall)
- Delay: Creates echoes of the sound
- Compressor: Evens out loud and quiet parts (used in podcasts)
- EQ: Boosts or cuts certain frequencies (more bass, less treble)
- Pitch Shift: Makes voice higher or lower

Effects can be:

- Chained: One after another (guitar -> distortion -> reverb -> amp)
- Real-time: Process audio live as it plays
- Offline: Process entire files for best quality

## How It Works

Audio effects modify sound in creative or corrective ways:

- Dynamics: Compressor, Limiter, Gate, Expander
- EQ: Parametric EQ, Graphic EQ, Filters
- Time-based: Reverb, Delay, Echo
- Modulation: Chorus, Flanger, Phaser, Tremolo
- Pitch: Pitch Shifter, Auto-Tune, Harmonizer
- Distortion: Overdrive, Fuzz, Saturation

## Properties

| Property | Summary |
|:-----|:--------|
| `Bypass` | Gets or sets whether the effect is bypassed (disabled). |
| `LatencySamples` | Gets the processing latency in samples. |
| `Mix` | Gets or sets the dry/wet mix (0.0 = dry only, 1.0 = wet only). |
| `Name` | Gets the name of this effect. |
| `Parameters` | Gets all adjustable parameters for this effect. |
| `SampleRate` | Gets the sample rate this effect operates at. |
| `TailSamples` | Gets the tail length in samples (how long the effect rings out after input stops). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameter(String)` | Gets a parameter value by name. |
| `Process(Tensor<>)` | Processes audio through the effect. |
| `ProcessInPlace(Span<>)` | Processes audio in-place for efficiency. |
| `ProcessSample()` | Processes a single sample (for real-time use). |
| `Reset` | Resets the effect's internal state. |
| `SetParameter(String,)` | Sets a parameter value by name. |

