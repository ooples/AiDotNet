---
title: "NoiseRobustASR<T>"
description: "Noise-Robust ASR: speech recognition in adverse acoustic conditions"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

Noise-Robust ASR: speech recognition in adverse acoustic conditions

## For Beginners

Noise-Robust ASR addresses speech recognition in challenging acoustic environments using multi-condition training and noise-aware preprocessing. The model is trained on diverse noisy conditions (babble, traffic, music, reverberation) with data aug...

## How It Works

**References:**

- Paper: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., OpenAI, 2023)

Noise-Robust ASR addresses speech recognition in challenging acoustic environments using multi-condition training and noise-aware preprocessing. The model is trained on diverse noisy conditions (babble, traffic, music, reverberation) with data augmentation including SpecAugment, speed perturbation, and room impulse response simulation. A noise estimation front-end preprocesses audio before the Conformer encoder, enabling robust transcription even at low SNR levels.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using noise-robust Conformer encoder with CTC decoding. |

