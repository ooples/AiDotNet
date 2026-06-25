---
title: "ParakeetRNNT<T>"
description: "Parakeet-RNNT: NVIDIA NeMo's 1.1B parameter Conformer with RNN-Transducer decoder."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

Parakeet-RNNT: NVIDIA NeMo's 1.1B parameter Conformer with RNN-Transducer decoder.

## For Beginners

Parakeet-RNNT pairs the same 24-layer Fast Conformer encoder as Parakeet-CTC with an RNN-Transducer decoder instead of CTC. The RNN-T decoder consists of a prediction network (LSTM) that models text history and a joint network that combines encode...

## How It Works

**References:**

- Model: "Parakeet-RNNT" (NVIDIA NeMo, 2024)

Parakeet-RNNT pairs the same 24-layer Fast Conformer encoder as Parakeet-CTC with an
RNN-Transducer decoder instead of CTC. The RNN-T decoder consists of a prediction network
(LSTM) that models text history and a joint network that combines encoder and prediction
outputs. This enables streaming-capable inference and generally better accuracy than CTC
alone since the decoder can model label dependencies. The joint network produces per-frame,
per-label logits for the RNN-T loss function.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Fast Conformer encoder + RNN-Transducer decoder. |

