---
title: "GroqWhisper<T>"
description: "Groq Whisper: hardware-accelerated Whisper inference"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ProprietaryAPI`

Groq Whisper: hardware-accelerated Whisper inference

## For Beginners

Groq Whisper provides Whisper model inference accelerated by Groq's LPU (Language Processing Unit) hardware. The LPU's deterministic execution model and high memory bandwidth enable real-time Whisper Large V3 inference at unprecedented speeds. Gro...

## How It Works

**References:**

- API: "Groq Whisper" (Groq, 2024)

Groq Whisper provides Whisper model inference accelerated by Groq's LPU (Language Processing Unit) hardware. The LPU's deterministic execution model and high memory bandwidth enable real-time Whisper Large V3 inference at unprecedented speeds. Groq achieves 189x real-time processing for Whisper Large V3, making it the fastest available Whisper API. The service supports all Whisper model sizes and languages through Groq's cloud API, with particular advantages for batch processing large audio datasets.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Whisper architecture on Groq LPU hardware. |

