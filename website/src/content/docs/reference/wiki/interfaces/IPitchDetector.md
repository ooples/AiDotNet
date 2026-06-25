---
title: "IPitchDetector<T>"
description: "Defines the contract for pitch (fundamental frequency) detection."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for pitch (fundamental frequency) detection.

## For Beginners

Pitch is what makes a note sound "high" or "low".

Technical definition:

- Pitch = perceived frequency of a sound
- F0 (fundamental frequency) = the lowest frequency component
- Measured in Hz (cycles per second)

Human voice pitch ranges:

- Bass: 80-300 Hz
- Baritone: 100-400 Hz
- Tenor: 130-500 Hz
- Alto: 175-700 Hz
- Soprano: 250-1000 Hz

Applications:

- Auto-tune / pitch correction (T-Pain effect)
- Music transcription (audio to sheet music)
- Karaoke scoring
- Speech therapy (monitoring pitch for dysphonia)
- Voice training for singing or public speaking
- Lie detection (pitch changes under stress)

Common algorithms:

- YIN: Fast, accurate for monophonic audio
- PYIN: Probabilistic YIN (handles uncertainty)
- CREPE: Neural network approach (most accurate)
- Autocorrelation: Classic signal processing method

## How It Works

Pitch detection finds the fundamental frequency (F0) of periodic signals.
This is essential for music analysis and speech processing.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxPitch` | Gets or sets the maximum detectable pitch in Hz. |
| `MinPitch` | Gets or sets the minimum detectable pitch in Hz. |
| `SampleRate` | Gets the sample rate this detector operates at. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectPitch(Tensor<>)` | Detects the pitch of an audio frame. |
| `DetectPitchWithConfidence(Tensor<>)` | Detects pitch with confidence score. |
| `ExtractDetailedPitchContour(Tensor<>,Int32)` | Extracts pitch contour with voicing information. |
| `ExtractPitchContour(Tensor<>,Int32)` | Extracts pitch contour from audio (F0 over time). |
| `GetCentsDeviation()` | Calculates cents deviation from nearest note. |
| `MidiToPitch(Double)` | Converts MIDI note number to pitch in Hz. |
| `PitchToMidi()` | Converts pitch in Hz to MIDI note number. |
| `PitchToNoteName()` | Gets the note name for a pitch. |

