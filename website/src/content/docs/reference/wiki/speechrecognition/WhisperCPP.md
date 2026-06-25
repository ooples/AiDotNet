---
title: "WhisperCPP<T>"
description: "Whisper.cpp: optimized C++ inference for Whisper models"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

Whisper.cpp: optimized C++ inference for Whisper models

## For Beginners

Whisper.cpp is a high-performance C++ implementation of OpenAI's Whisper model optimized for CPU inference. It uses GGML tensor library with quantization support (Q4, Q5, Q8) for reduced memory and faster processing. The implementation supports Ap...

## How It Works

**References:**

- Software: "whisper.cpp" (Gerganov, 2022-2025)

Whisper.cpp is a high-performance C++ implementation of OpenAI's Whisper model optimized for CPU inference. It uses GGML tensor library with quantization support (Q4, Q5, Q8) for reduced memory and faster processing. The implementation supports Apple Silicon acceleration (Metal, Core ML, ANE), x86 AVX/AVX2/AVX-512, and ARM NEON. Whisper.cpp achieves 10-20x speedup over the original Python implementation and enables real-time transcription on consumer hardware without GPU.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Whisper architecture with optimized inference. |

