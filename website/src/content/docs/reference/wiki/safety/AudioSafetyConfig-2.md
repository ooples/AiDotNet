---
title: "AudioSafetyConfig"
description: "Configuration for audio safety modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety`

Configuration for audio safety modules.

## For Beginners

These settings control how audio content is checked for safety.
Deepfake detection identifies cloned or synthetic voices, and toxic speech detection
catches harmful spoken content.

## How It Works

**References:**

- SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
- AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)

## Properties

| Property | Summary |
|:-----|:--------|
| `DeepfakeDetection` | Gets or sets whether audio deepfake detection is enabled. |
| `SampleRate` | Gets or sets the expected sample rate in Hz. |
| `ToxicSpeechDetection` | Gets or sets whether toxic speech detection is enabled. |

