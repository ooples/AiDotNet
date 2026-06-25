---
title: "PerturbationVoiceProtector<T>"
description: "Protects voice recordings against cloning by adding imperceptible adversarial perturbations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Audio`

Protects voice recordings against cloning by adding imperceptible adversarial perturbations.

## For Beginners

This is like adding an invisible "anti-copying" pattern to a voice
recording. Humans can't hear the difference, but if someone tries to clone the voice using
AI, the protection disrupts the cloning process. Think of it like a DRM for your voice.

## How It Works

Adds carefully crafted perturbations to the audio that disrupt voice cloning systems while
remaining inaudible to human listeners. The perturbations target the frequency bands most
important for speaker embedding extraction (typically 300-3400 Hz for formants) and add
noise shaped to psychoacoustic masking thresholds.

**References:**

- SafeSpeech: SPEC perturbation framework against voice cloning (2025, arxiv:2504.09839)
- Adversarial examples for speech recognition (Carlini & Wagner, 2018)
- Voice protection via perturbation (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PerturbationVoiceProtector(Double,Int32)` | Initializes a new perturbation-based voice protector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAudio(Vector<>,Int32)` |  |
| `HashInt(Int32)` | Integer hash using MurmurHash3 finalizer for deterministic pseudo-random perturbation. |
| `ProtectAudio(Vector<>,Int32)` | Applies voice protection perturbations and returns protected audio with safety findings. |

