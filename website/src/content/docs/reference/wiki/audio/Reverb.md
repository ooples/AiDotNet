---
title: "Reverb<T>"
description: "Algorithmic reverb effect using Schroeder-Moorer structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Effects`

Algorithmic reverb effect using Schroeder-Moorer structure.

## For Beginners

Reverb makes things sound like they're in a room!

What reverb is:

- Sound bounces off walls, ceiling, floor
- These reflections blend together
- Creates a sense of space and depth

Key parameters:

- Room Size: How big the virtual room is
- Decay Time: How long reverb rings (small room = short, hall = long)
- Pre-delay: Time before reverb starts (sense of room distance)
- Damping: How much high frequencies are absorbed
- Wet/Dry: Balance between original and reverb signal

Types of reverb sounds:

- Room: Small, intimate (0.2-0.5s decay)
- Hall: Large concert hall (1-2s decay)
- Cathedral: Huge, ethereal (3-6s decay)
- Plate: Artificial, bright (classic 80s sound)
- Spring: Metallic, twangy (guitar amps)

This implementation uses:

- Allpass filters for diffusion (smears the early reflections)
- Comb filters for resonance (creates the decay tail)
- Low-pass filter for damping (natural high-frequency absorption)

## How It Works

Reverb simulates the acoustic reflections of a physical space,
adding depth and ambience to dry recordings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Reverb(Int32,Double,Double,Double,Double,Double,Double)` | Creates a reverb effect with room-style defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `TailSamples` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnParameterChanged(String,)` |  |
| `ProcessSampleInternal()` |  |
| `Reset` |  |

