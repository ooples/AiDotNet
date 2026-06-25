---
title: "ESPnetASR<T>"
description: "ESPnet ASR: end-to-end speech processing toolkit models"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Robust`

ESPnet ASR: end-to-end speech processing toolkit models

## For Beginners

ESPnet provides state-of-the-art end-to-end speech processing models with a focus on reproducibility and benchmarking. ESPnet2 ASR models support Conformer, Branchformer, and E-Branchformer encoders with CTC, attention, and transducer decoders. Th...

## How It Works

**References:**

- Paper: "ESPnet: End-to-End Speech Processing Toolkit" (Watanabe et al., 2018, updated 2024)

ESPnet provides state-of-the-art end-to-end speech processing models with a focus on reproducibility and benchmarking. ESPnet2 ASR models support Conformer, Branchformer, and E-Branchformer encoders with CTC, attention, and transducer decoders. The toolkit includes joint CTC/attention training, language model shallow fusion, and multi-task learning. ESPnet models consistently achieve top results on standard benchmarks including LibriSpeech, AISHELL, and CommonVoice.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using ESPnet's Conformer encoder with joint CTC/attention. |

