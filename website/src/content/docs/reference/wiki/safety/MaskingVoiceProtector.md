---
title: "MaskingVoiceProtector<T>"
description: "Protects voice recordings against cloning using psychoacoustic masking — adding noise that is hidden beneath audible content but disrupts speaker embedding extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Protects voice recordings against cloning using psychoacoustic masking — adding noise
that is hidden beneath audible content but disrupts speaker embedding extraction.

## For Beginners

Our ears have blind spots — when a loud sound is playing, we can't
hear quiet sounds nearby. This module exploits those blind spots to hide anti-cloning noise
where your ears can't detect it, but AI cloning systems can "hear" and get confused by.

## How It Works

Applies frequency-domain noise shaped to the psychoacoustic masking curve of the audio.
The masking curve determines the threshold below which noise is inaudible to humans;
this protector adds maximum disruption noise just below this threshold. The disruption
specifically targets the frequency bands used by speaker verification systems.

**References:**

- VocalCrypt: Pseudo-timbre jamming for voice protection (2025, arxiv:2502.10329)
- Psychoacoustic masking models for audio steganography (2023)
- MPEG psychoacoustic model for perceptual coding (ISO/IEC 11172-3)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaskingVoiceProtector(Double,Int32)` | Initializes a new masking-based voice protector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAbsoluteThreshold(Double)` | Simplified absolute hearing threshold (ISO 226). |
| `ComputeMaskingCurve(Double[],Int32,Int32,Int32)` | Computes a simplified psychoacoustic masking curve. |
| `EvaluateAudio(Vector<>,Int32)` |  |
| `ProtectAudio(Vector<>,Int32)` | Applies psychoacoustic masking protection and returns protected audio. |

