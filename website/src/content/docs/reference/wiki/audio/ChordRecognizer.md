---
title: "ChordRecognizer<T>"
description: "Recognizes chords from audio using chromagram analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Recognizes chords from audio using chromagram analysis.

## For Beginners

A chord is a group of notes played together.
Different chords have different "shapes" in their chroma patterns:

- C major (C-E-G) has energy in bins 0, 4, and 7
- C minor (C-Eb-G) has energy in bins 0, 3, and 7

This algorithm looks at the audio and matches it against known chord patterns
to tell you what chord is playing at each moment.

Usage:

## How It Works

Chord recognition works by comparing chromagram patterns against known chord templates.
Supports major, minor, diminished, augmented, dominant 7th, major 7th, and minor 7th chords.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChordRecognizer(ChordRecognizerOptions)` | Creates a new chord recognizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Recognize(Tensor<>)` | Recognizes chords in the audio. |
| `Recognize(Vector<>)` | Recognizes chords in the audio. |

