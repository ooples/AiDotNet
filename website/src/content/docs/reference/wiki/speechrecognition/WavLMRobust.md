---
title: "WavLMRobust<T>"
description: "WavLM-Robust: noise-robust speech recognition via denoising pre-training"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

WavLM-Robust: noise-robust speech recognition via denoising pre-training

## For Beginners

WavLM-Robust is a specialized ASR model leveraging WavLM's unique denoising pre-training for robust speech recognition. During pre-training, overlapping utterances from different speakers are used as input, and the model learns to denoise and sepa...

## How It Works

**References:**

- Paper: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (Chen et al., Microsoft, 2022)

WavLM-Robust is a specialized ASR model leveraging WavLM's unique denoising pre-training for robust speech recognition. During pre-training, overlapping utterances from different speakers are used as input, and the model learns to denoise and separate while predicting masked tokens. This produces representations inherently robust to noise, reverberation, and overlapping speech. Fine-tuned with CTC for ASR, WavLM-Robust achieves strong performance on noisy benchmarks like CHiME-4 and VoiceBank-DEMAND.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using WavLM's denoising-pretrained encoder with CTC. |

