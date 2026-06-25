---
title: "ChromaExtractor<T>"
description: "Extracts chromagram (pitch class profile) features from audio signals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Features`

Extracts chromagram (pitch class profile) features from audio signals.

## For Beginners

In Western music, there are 12 notes that repeat in each octave.
A C note at 262 Hz and a C note at 524 Hz are both "C" - they're the same pitch class.

A chromagram collapses all octaves together, showing how much energy is in each of the 12 notes:

- Index 0: C (do)
- Index 1: C#/Db
- Index 2: D (re)
- ...and so on through B

This is useful for:

- Chord recognition (chords have characteristic chroma patterns)
- Key detection (which notes are emphasized in the music)
- Music similarity (songs in the same key have similar chromagrams)
- Cover song detection

Usage:

## How It Works

A chromagram represents the energy of the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
regardless of octave. It's particularly useful for music analysis tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChromaExtractor(ChromaOptions)` | Initializes a new chroma feature extractor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Extract(Tensor<>)` |  |
| `GetDominantPitchClass([])` | Gets the dominant pitch class for a chroma vector. |
| `GetPitchClassName(Int32)` | Gets the pitch class name for an index (0-11). |

