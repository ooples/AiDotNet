---
title: "WavLMASR<T>"
description: "WavLM ASR: speech pre-training with denoising and attention"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Foundation`

WavLM ASR: speech pre-training with denoising and attention

## For Beginners

WavLM extends HuBERT with: (1) gated relative position bias in self-attention for better sequence modeling; (2) denoising pre-training using overlapping utterances to learn robust representations in noisy conditions. The model is pre-trained on 94...

## How It Works

**References:**

- Paper: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (Chen et al., Microsoft, 2022)

WavLM extends HuBERT with: (1) gated relative position bias in self-attention for better sequence modeling; (2) denoising pre-training using overlapping utterances to learn robust representations in noisy conditions. The model is pre-trained on 94k hours of speech data. WavLM achieves state-of-the-art on the SUPERB benchmark across ASR, speaker verification, separation, and other tasks.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using WavLM's denoised SSL encoder with CTC decoding. |

