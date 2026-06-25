---
title: "CTCSegmentation<T>"
description: "CTC Segmentation: forced alignment using CTC posteriors"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.CTCVariants`

CTC Segmentation: forced alignment using CTC posteriors

## For Beginners

CTC Segmentation uses CTC posterior probabilities for forced alignment of speech to text. Given pre-computed CTC posteriors and a text transcript, the algorithm finds the optimal alignment between frames and characters using dynamic programming. T...

## How It Works

**References:**

- Paper: "CTC-based Audio Segmentation and Alignment" (Kurzinger et al., 2020)

CTC Segmentation uses CTC posterior probabilities for forced alignment of speech to text. Given pre-computed CTC posteriors and a text transcript, the algorithm finds the optimal alignment between frames and characters using dynamic programming. This enables precise word and phoneme boundaries without requiring an explicit alignment model. Applications include creating training data from long-form audio, subtitle synchronization, and corpus alignment for TTS training.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using CTC posteriors with forced alignment. |

