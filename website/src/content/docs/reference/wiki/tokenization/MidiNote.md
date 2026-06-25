---
title: "MidiNote"
description: "Represents a MIDI note event with timing, pitch, and velocity information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.Specialized`

Represents a MIDI note event with timing, pitch, and velocity information.

## For Beginners

A MIDI note is like a single key press on a piano:

- StartTick: When in the song the note starts (like a timestamp)
- Duration: How long the key is held down
- Pitch: Which key is pressed (60 = middle C, higher = higher notes)
- Velocity: How hard the key is pressed (louder = higher velocity)

Example: A middle C played for one beat at medium volume might be:
StartTick=0, Duration=480, Pitch=60, Velocity=64

## Properties

| Property | Summary |
|:-----|:--------|
| `Duration` | Gets or sets the duration of the note in MIDI ticks. |
| `Pitch` | Gets or sets the pitch (0-127) representing the musical note. |
| `StartTick` | Gets or sets the start tick of the note in MIDI ticks. |
| `Velocity` | Gets or sets the velocity (0-127) representing the note intensity. |

