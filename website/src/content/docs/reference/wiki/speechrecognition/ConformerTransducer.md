---
title: "ConformerTransducer<T>"
description: "Conformer-Transducer: Conformer encoder with RNN-T/TDT decoder for streaming ASR."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Conformer-Transducer: Conformer encoder with RNN-T/TDT decoder for streaming ASR.

## For Beginners

Combines the Conformer encoder's strong acoustic modeling with the RNN-T decoder's streaming capability. The prediction network maintains output history, the joint network combines encoder and prediction states, and the model autoregressively emit...

## How It Works

**References:**

- Paper: "Conformer: Convolution-augmented Transformer" (Gulati et al., 2020) + RNN-T (Graves, 2012)

Combines the Conformer encoder's strong acoustic modeling with the RNN-T decoder's
streaming capability. The prediction network maintains output history, the joint network
combines encoder and prediction states, and the model autoregressively emits tokens
frame by frame. Used in production at Google (Pixel phones) and NVIDIA Riva.

## Methods

| Method | Summary |
|:-----|:--------|
| `GreedyDecodeWithConfidence(Tensor<>)` | Greedy decoding on output logits. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Conformer encoder + RNN-T decoder. |

