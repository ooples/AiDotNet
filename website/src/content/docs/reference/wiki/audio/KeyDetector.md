---
title: "KeyDetector<T>"
description: "Detects the musical key of audio using chromagram analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Detects the musical key of audio using chromagram analysis.

## For Beginners

The "key" of a song tells you which notes are
emphasized and how the music "feels." For example:

- C major: Bright, happy sound (uses mainly white keys on piano)
- A minor: Sadder, darker sound (also mainly white keys, but starts on A)

Knowing the key helps with:

- Playing along with songs
- Transposing music to different keys
- Understanding the harmonic structure

Usage:

## How It Works

Key detection uses the Krumhansl-Kessler key profiles to match average
chroma features against major and minor key templates. This provides
the most likely key and its relative minor/major.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KeyDetector(KeyDetectorOptions)` | Creates a new key detector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Detect(Tensor<>)` | Detects the musical key of the audio. |
| `Detect(Vector<>)` | Detects the musical key of the audio. |
| `DetectAll(Tensor<>)` | Gets all key detection results ranked by correlation. |

