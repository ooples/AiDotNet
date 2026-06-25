---
title: "VoiceProtectorBase<T>"
description: "Abstract base class for voice protection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Audio`

Abstract base class for voice protection modules.

## For Beginners

This base class provides common code for all voice protectors.
Each protector type extends this and adds its own way of making voice recordings
resistant to AI voice cloning.

## How It Works

Provides shared infrastructure for voice protectors including sample rate
configuration and signal-to-noise ratio monitoring. Concrete implementations
provide the actual protection technique (perturbation, watermark, masking).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VoiceProtectorBase(Int32)` | Initializes the voice protector base. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Vector<>)` |  |
| `ProtectVoice(Vector<>,Int32)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultSampleRate` | The default sample rate for audio processing. |

