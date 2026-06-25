---
title: "OWSM<T>"
description: "OWSM: Open Whisper-Style Speech Model"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Multilingual`

OWSM: Open Whisper-Style Speech Model

## For Beginners

OWSM is an open-source reproduction of Whisper's training approach using publicly available data. It uses ESPnet's encoder-decoder architecture with a Conformer encoder and Transformer decoder, trained on a curated collection of open speech datase...

## How It Works

**References:**

- Paper: "OWSM: Open Whisper-style Speech Models" (Peng et al., CMU, 2024)

OWSM is an open-source reproduction of Whisper's training approach using publicly available data. It uses ESPnet's encoder-decoder architecture with a Conformer encoder and Transformer decoder, trained on a curated collection of open speech datasets. OWSM supports ASR, translation, and language identification. Being fully open-source (data, code, and model), it enables reproducible multilingual ASR research.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using OWSM's open-source Whisper-style encoder-decoder. |

