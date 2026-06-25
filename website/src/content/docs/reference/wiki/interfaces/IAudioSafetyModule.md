---
title: "IAudioSafetyModule<T>"
description: "Interface for safety modules that operate on audio content."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for safety modules that operate on audio content.

## For Beginners

Audio safety modules check sound content for problems like
fake voices (deepfakes), hateful speech, and cloned voices. They help protect against
voice impersonation and audio-based fraud.

## How It Works

Audio safety modules analyze audio waveforms for safety risks such as deepfake voices,
toxic speech, and AI-generated audio. They can also detect embedded watermarks.

**References:**

- SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
- AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)
- VoiceRadar: Voice deepfake detection (NDSS 2025)

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAudio(Vector<>,Int32)` | Evaluates the given audio waveform for safety and returns any findings. |

