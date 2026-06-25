---
title: "FunASRNano<T>"
description: "FunASR-Nano: ultra-lightweight on-device ASR"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.AlibabaASR`

FunASR-Nano: ultra-lightweight on-device ASR

## For Beginners

FunASR-Nano is an ultra-lightweight ASR model designed for on-device deployment. It uses a compact Conformer encoder (256-dim, 6 layers) with aggressive subsampling and a CTC decoder. Knowledge distillation from Paraformer-Large provides strong pe...

## How It Works

**References:**

- Model: "FunASR-Nano" (Alibaba FunASR, 2024)

FunASR-Nano is an ultra-lightweight ASR model designed for on-device deployment. It uses a compact Conformer encoder (256-dim, 6 layers) with aggressive subsampling and a CTC decoder. Knowledge distillation from Paraformer-Large provides strong performance despite the small model size. Optimized for mobile and edge devices with under 50MB model size and real-time inference on CPU.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using the ultra-compact FunASR-Nano encoder with CTC. |

