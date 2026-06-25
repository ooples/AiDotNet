---
title: "SambaASR<T>"
description: "Samba-ASR: Mamba-based state-space model for speech recognition"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.LLMIntegrated`

Samba-ASR: Mamba-based state-space model for speech recognition

## For Beginners

Samba-ASR applies the Mamba selective state-space model architecture to speech recognition. Unlike Transformer-based ASR with quadratic attention complexity, Mamba's linear-time SSM layers process sequences efficiently. The model uses selective sc...

## How It Works

**References:**

- Paper: "An Exploration of State Space Models and Mamba for Speech Recognition" (Miyazaki et al., 2024)

Samba-ASR applies the Mamba selective state-space model architecture to speech recognition. Unlike Transformer-based ASR with quadratic attention complexity, Mamba's linear-time SSM layers process sequences efficiently. The model uses selective scan mechanisms that dynamically filter speech features based on content, achieving competitive WER with significantly lower computational cost for long-form audio processing.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Mamba SSM encoder with CTC decoding. |

