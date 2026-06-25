---
title: "WhisperTimestamped<T>"
description: "WhisperTimestamped: Cross-attention-based word-level timestamps for Whisper."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.WhisperFamily`

WhisperTimestamped: Cross-attention-based word-level timestamps for Whisper.

## For Beginners

WhisperTimestamped extracts word-level timestamps from Whisper's cross-attention weights without any additional training or model modification. The key insight: cross-attention weights between decoder tokens and encoder frames reveal temporal alig...

## How It Works

**References:**

- Paper: "whisper-timestamped: Word-level timestamps for Whisper" (Louradour, 2023)

WhisperTimestamped extracts word-level timestamps from Whisper's cross-attention weights
without any additional training or model modification. The key insight: cross-attention
weights between decoder tokens and encoder frames reveal temporal alignment. For each
generated word token, the algorithm finds the encoder frame with maximum cross-attention
weight, then converts frame index to timestamp. Dynamic Time Warping (DTW) refines the
alignment to ensure monotonicity. The method works on any Whisper model variant and adds
negligible overhead to inference.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio with cross-attention-based word timestamps. |

