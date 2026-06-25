---
title: "BeatTracker<T>"
description: "Extracts beat and tempo information from audio."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Extracts beat and tempo information from audio.

## For Beginners

The "beat" is what you tap your foot to when
listening to music. This algorithm finds:

- Tempo: How fast the music is (measured in BPM - beats per minute)
- Beat times: Exactly when each beat occurs in the song

For example, a typical pop song is around 120 BPM, meaning there are
120 beats per minute (2 beats per second).

Usage:

## How It Works

Beat tracking involves detecting the tempo (beats per minute) and the
specific times when beats occur in the audio. This is fundamental for
music synchronization and rhythm analysis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BeatTracker(BeatTrackerOptions)` | Creates a new beat tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxBPM` | Gets the maximum BPM for beat detection. |
| `MinBPM` | Gets the minimum BPM for beat detection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Track(Tensor<>)` | Tracks beats in the audio. |
| `Track(Vector<>)` | Tracks beats in the audio. |

