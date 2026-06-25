---
title: "TranscriptionToxicityDetector<T>"
description: "Detects toxic content in audio by analyzing acoustic features indicative of aggressive speech."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Detects toxic content in audio by analyzing acoustic features indicative of aggressive speech.

## For Beginners

While text-based toxicity detection catches harmful words, this module
analyzes the sound itself — shouting, aggressive tone, and speech patterns associated with
hateful content can be detected from audio features alone.

## How It Works

This module analyzes acoustic features (energy, pitch variation, speaking rate) that correlate
with aggressive or hateful speech patterns. It acts as an audio-level complement to text-based
toxicity detection by catching patterns that transcription-based approaches might miss.

**Detection approach:**

1. Energy analysis — sudden loud bursts can indicate shouting/aggression
2. Speaking rate estimation — very rapid speech can correlate with agitation
3. Dynamic range — extreme variation in volume may indicate emotional distress
4. Short-term energy variance — high variance suggests shouting patterns
5. Zero-crossing rate — correlates with voiced/unvoiced speech characteristics

**References:**

- Multimodal toxicity detection: combining audio + text signals (ACL 2024)
- Speech emotion recognition for content moderation (INTERSPEECH 2024)
- Hate speech detection in speech: challenges and approaches (2024)
- Acoustic correlates of aggression in speech (Journal of Phonetics, 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TranscriptionToxicityDetector(Double,Int32)` | Initializes a new transcription toxicity detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EstimateAcousticToxicity(TranscriptionToxicityDetector<>.AcousticFeatures)` | Estimates acoustic toxicity from features using a weighted heuristic model. |
| `EvaluateAudio(Vector<>,Int32)` |  |

