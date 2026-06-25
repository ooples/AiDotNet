---
title: "AudioSafetyResult"
description: "Detailed result from audio safety evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Detailed result from audio safety evaluation.

## For Beginners

AudioSafetyResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `DeepfakeScore` | Deepfake probability score (0.0 = authentic, 1.0 = fake). |
| `IsSafe` | Whether the audio is safe overall. |
| `SampleRate` | Detected sample rate of the audio. |
| `ToxicityScore` | Toxicity score (0.0 = safe, 1.0 = maximally toxic). |
| `WatermarkDetected` | Whether a watermark was detected. |

