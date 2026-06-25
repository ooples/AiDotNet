---
title: "VoiceprintDeepfakeDetector<T>"
description: "Detects deepfake audio by analyzing speaker voiceprint consistency throughout the recording."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Detects deepfake audio by analyzing speaker voiceprint consistency throughout the recording.

## For Beginners

Each person's voice has unique characteristics — like a fingerprint.
Real speech keeps these characteristics consistent. Voice cloning often produces subtle
inconsistencies that this module detects by comparing "voice fingerprints" from different
parts of the recording.

## How It Works

Extracts short-term speaker embeddings from overlapping frames of the audio and measures
consistency across the recording. Real speech has stable speaker characteristics (pitch
contour, formant patterns, spectral envelope) while cloned/synthesized voices often show
temporal inconsistencies in these features — either too stable (robotic) or with sudden
jumps (spliced segments).

**References:**

- VoiceRadar: Voice deepfake detection framework (NDSS 2025)
- SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
- Voice cloning detection via speaker verification mismatch (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoiceprintDeepfakeDetector(Double,Int32)` | Initializes a new voiceprint deepfake detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAudio(Vector<>,Int32)` |  |

