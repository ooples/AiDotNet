---
title: "IKeyDetector<T>"
description: "Interface for musical key detection models that identify the key and mode of music."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for musical key detection models that identify the key and mode of music.

## For Beginners

The musical key is like the "home base" of a song.

What is a key?

- Every song has a central note that feels like "home"
- The key tells you which note that is and whether it's major (happy) or minor (sad)
- "C major" means C is home and it sounds happy
- "A minor" means A is home and it sounds sad/dark

How key detection works:

1. Audio is analyzed to find which notes are used most
2. This is compared to key profiles (templates of note usage)
3. The best-matching key is selected

Why it matters:

- DJ mixing (match keys for smooth transitions)
- Music recommendation (similar keys = similar feel)
- Music production (know what key to write melodies in)
- Transposition (shifting a song to a different key)

Related concepts:

- Relative keys: Am is the relative minor of C major (same notes)
- Parallel keys: C major and C minor (same root, different mode)

## How It Works

Key detection identifies the musical key (e.g., C major, A minor) of a piece of music.
The key defines the central note (tonic) and scale (major/minor) that the music is based on.

## Properties

| Property | Summary |
|:-----|:--------|
| `SampleRate` | Gets the expected sample rate for input audio. |
| `SupportedKeys` | Gets the list of keys this model can detect. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` | Detects the musical key of audio. |
| `DetectAsync(Tensor<>,CancellationToken)` | Detects the musical key asynchronously. |
| `GetCamelotNotation(String)` | Gets the Camelot wheel notation for a key. |
| `GetCompatibleKeys(String)` | Finds compatible keys for mixing. |
| `GetKeyProbabilities(Tensor<>)` | Gets key probabilities for all possible keys. |
| `GetRelativeKey(String)` | Gets the relative major/minor key. |
| `TrackKeyChanges(Tensor<>,Double)` | Tracks key changes over time within a piece. |

