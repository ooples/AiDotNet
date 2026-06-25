---
title: "ConvTransformer<T>"
description: "Convolution-augmented Transformer for ASR (pre-Conformer architecture)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Convolution-augmented Transformer for ASR (pre-Conformer architecture).

## For Beginners

A precursor to the Conformer that adds convolution blocks before or after each Transformer layer. Unlike Conformer's interleaved design, ConvTransformer uses convolution as a separate preprocessing stage, which provides local feature enhancement b...

## How It Works

**References:**

- Paper: "Convolution-Augmented Transformer for Speech Recognition" (2019)

A precursor to the Conformer that adds convolution blocks before or after each
Transformer layer. Unlike Conformer's interleaved design, ConvTransformer uses
convolution as a separate preprocessing stage, which provides local feature enhancement
before the global self-attention. This architecture demonstrated the importance of
combining convolution with attention for speech.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using convolution-augmented Transformer encoder. |

