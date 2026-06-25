---
title: "AcousticToxicityDetector<T>"
description: "Detects toxic/aggressive speech patterns directly from acoustic features without transcription."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Detects toxic/aggressive speech patterns directly from acoustic features without transcription.

## For Beginners

You can often tell someone is angry or threatening just from the
tone of their voice — even if you don't understand the language. This module does the same:
it analyzes voice patterns like loudness changes, pitch, and speaking rate to detect
aggressive or hostile speech.

## How It Works

Analyzes prosodic and spectral features that correlate with aggression, shouting, and hostile
intent. Aggressive speech has characteristic acoustic signatures: elevated pitch, high energy
variance, rapid pitch changes, and spectral centroid shifts. This approach works regardless
of language and catches toxicity even when words are unintelligible.

**References:**

- Taxonomy of speech generator harms including swatting attacks (2024, arxiv:2402.01708)
- Acoustic emotion recognition using prosodic and spectral features (2023)
- Paralinguistic analysis for aggression detection (INTERSPEECH 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AcousticToxicityDetector(Double,Int32)` | Initializes a new acoustic toxicity detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAudio(Vector<>,Int32)` |  |

