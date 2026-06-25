---
title: "IAudioDeepfakeDetector<T>"
description: "Interface for audio deepfake and voice cloning detection modules."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Safety.Audio`

Interface for audio deepfake and voice cloning detection modules.

## For Beginners

An audio deepfake detector checks if a voice recording is
real or AI-generated. It can detect cloned voices, synthesized speech, and voice
conversion attacks by analyzing subtle patterns in the audio signal.

## How It Works

Audio deepfake detectors analyze audio waveforms for signs of AI-generated speech,
voice cloning, or voice conversion. Approaches include spectral analysis of
mel spectrograms, speaker embedding verification, and watermark detection.

**References:**

- SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
- VoiceRadar: Voice deepfake detection framework (NDSS 2025)
- AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDeepfakeScore(Vector<>,Int32)` | Gets the deepfake probability score for the given audio (0.0 = authentic, 1.0 = fake). |

