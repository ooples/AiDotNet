---
title: "IBeatTracker<T>"
description: "Interface for beat tracking models that detect tempo and beat positions in audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for beat tracking models that detect tempo and beat positions in audio.

## For Beginners

Beat tracking is like tapping your foot to music - finding the pulse.

How it works:

1. Audio is analyzed for rhythmic events (drum hits, bass notes, etc.)
2. Periodicity detection finds the most likely beat period
3. Beat positions are refined to align with actual events

Common use cases:

- Music tempo detection ("this song is 120 BPM")
- DJ software (beat matching between songs)
- Music games (rhythm games like Guitar Hero)
- Audio visualization (beat-synced lights)
- Music production (quantizing to the beat)

Key concepts:

- BPM (Beats Per Minute): The tempo or speed of the music
- Downbeat: The first beat of a measure (often emphasized)
- Beat phase: Where in the beat cycle we are

## How It Works

Beat tracking analyzes audio to find the rhythmic pulse (beats) and estimate the tempo
(beats per minute). This is fundamental to music information retrieval and enables
beat-synchronized processing like auto-DJ mixing and rhythmic visualization.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxBPM` | Gets the maximum detectable BPM. |
| `MinBPM` | Gets the minimum detectable BPM. |
| `SampleRate` | Gets the expected sample rate for input audio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOnsetStrength(Tensor<>)` | Computes onset strength envelope for visualization or custom processing. |
| `DetectDownbeats(Tensor<>,BeatTrackingResult<>)` | Detects downbeat positions (first beat of each measure). |
| `EstimateTempo(Tensor<>)` | Estimates tempo without detecting individual beat positions. |
| `GetTempoHypotheses(Tensor<>,Int32)` | Gets multiple tempo hypotheses with confidence scores. |
| `Track(Tensor<>)` | Detects tempo and beat positions in audio. |
| `TrackAsync(Tensor<>,CancellationToken)` | Detects tempo and beat positions asynchronously. |

