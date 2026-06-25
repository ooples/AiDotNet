---
title: "AudioDeepfakeDetectorBase<T>"
description: "Abstract base class for audio deepfake detection modules."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Safety.Audio`

Abstract base class for audio deepfake detection modules.

## For Beginners

This base class provides common code for all audio deepfake
detectors. Each detector type extends this and adds its own way of detecting
AI-generated or cloned voices.

## How It Works

Provides shared infrastructure for audio deepfake detectors including sample rate
configuration and common spectral analysis utilities. Concrete implementations
provide the actual detection algorithm (spectral, voiceprint, watermark).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AudioDeepfakeDetectorBase(Int32)` | Initializes the audio deepfake detector base. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDeepfakeScore(Vector<>,Int32)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultSampleRate` | The default sample rate for audio processing. |

