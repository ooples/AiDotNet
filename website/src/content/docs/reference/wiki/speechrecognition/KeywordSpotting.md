---
title: "KeywordSpotting<T>"
description: "Keyword Spotting: lightweight wake-word and command detection"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Specialized`

Keyword Spotting: lightweight wake-word and command detection

## For Beginners

Keyword Spotting provides lightweight, always-on detection of specific keywords and voice commands. The model uses a compact encoder (4-layer Conformer with reduced dimensions) optimized for low-power continuous inference. Instead of full CTC voca...

## How It Works

**References:**

- Paper: "Streaming Keyword Spotting on Mobile Devices" (2023)

Keyword Spotting provides lightweight, always-on detection of specific keywords and voice commands. The model uses a compact encoder (4-layer Conformer with reduced dimensions) optimized for low-power continuous inference. Instead of full CTC vocabulary decoding, the model outputs confidence scores for a small set of predefined keywords. The model runs in streaming mode with a sliding window, using a fixed-point quantized encoder for minimal power consumption on edge devices.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Detects keywords using a compact streaming Conformer encoder. |

