---
title: "SPIRAL<T>"
description: "SPIRAL: self-supervised pre-training for speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

SPIRAL: self-supervised pre-training for speech recognition

## For Beginners

SPIRAL learns perturbation-invariant speech representations through self-supervised pre-training. The model uses a teacher-student framework where the student must produce similar representations for different augmented views of the same speech. D...

## How It Works

**References:**

- Paper: "SPIRAL: Self-supervised Perturbation-Invariant Representation Learning for Speech Pre-Training" (Huang et al., 2022)

SPIRAL learns perturbation-invariant speech representations through self-supervised pre-training. The model uses a teacher-student framework where the student must produce similar representations for different augmented views of the same speech. Data augmentations include speed perturbation, SpecAugment, and noise addition. The invariance objective encourages robust representations that transfer well to downstream ASR tasks.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using SPIRAL's perturbation-invariant SSL encoder with CTC. |

