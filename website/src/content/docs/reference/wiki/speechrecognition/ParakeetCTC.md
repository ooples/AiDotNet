---
title: "ParakeetCTC<T>"
description: "Parakeet-CTC: NVIDIA NeMo's 1.1B parameter Conformer-CTC with word-piece tokenization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Parakeet-CTC: NVIDIA NeMo's 1.1B parameter Conformer-CTC with word-piece tokenization.

## For Beginners

Parakeet-CTC is NVIDIA's state-of-the-art English ASR model built on the NeMo framework. It uses a 24-layer Fast Conformer encoder (optimized Conformer with multi-scale convolutions and 8x subsampling) with a CTC decoder head. Key innovations: Sen...

## How It Works

**References:**

- Model: "Parakeet" (NVIDIA NeMo, 2024)

Parakeet-CTC is NVIDIA's state-of-the-art English ASR model built on the NeMo framework.
It uses a 24-layer Fast Conformer encoder (optimized Conformer with multi-scale convolutions
and 8x subsampling) with a CTC decoder head. Key innovations: SentencePiece word-piece
tokenization (1024 tokens) instead of character-level, enabling better modeling of English
morphology. The model achieves <3% WER on LibriSpeech test-clean without LM rescoring.
Fast Conformer uses grouped multi-head attention and depthwise-separable convolutions for
efficient processing.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Parakeet's Fast Conformer encoder with CTC decoding. |

