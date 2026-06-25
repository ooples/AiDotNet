---
title: "IChordRecognizer<T>"
description: "Interface for chord recognition models that identify musical chords in audio."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for chord recognition models that identify musical chords in audio.

## For Beginners

Chord recognition is like having a musician listen to a song
and tell you what chords are being played.

How it works:

1. Audio is converted to a chromagram (12 pitch classes)
2. The pitch content is compared to known chord templates
3. The best-matching chord is selected for each time frame

What are chords?

- A chord is multiple notes played together
- "C major" = C + E + G notes together
- "A minor" = A + C + E notes together
- The chord creates the harmony of the music

Common use cases:

- Learning songs (getting chord charts automatically)
- Music production (analyzing harmony)
- Music generation (understanding structure)
- Cover song detection (comparing harmonic content)

## How It Works

Chord recognition analyzes audio to identify the musical chords being played.
This involves detecting the simultaneous notes and classifying them into
standard chord types (major, minor, seventh, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedChordTypes` | Gets the list of chord types this model can recognize. |
| `TimeResolution` | Gets the time resolution for chord detection in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractChromagram(Tensor<>)` | Extracts chromagram features from audio. |
| `GetChordNotes(String)` | Converts a chord symbol to its component notes. |
| `GetChordProbabilities(Tensor<>)` | Gets chord probabilities for each time frame. |
| `Recognize(Tensor<>)` | Recognizes chords in audio. |
| `RecognizeAsync(Tensor<>,CancellationToken)` | Recognizes chords asynchronously. |

