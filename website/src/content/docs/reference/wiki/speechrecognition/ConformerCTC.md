---
title: "ConformerCTC<T>"
description: "Conformer-CTC: Conformer encoder with CTC-only decoding (no external decoder)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.ConformerFamily`

Conformer-CTC: Conformer encoder with CTC-only decoding (no external decoder).

## For Beginners

This is the CTC-only variant of Conformer: the encoder output is projected to vocabulary size and decoded with CTC greedy search. Unlike the attention-decoder or transducer variants, CTC decoding is fully non-autoregressive and very fast.

## How It Works

**References:**

- Paper: "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020)

This is the CTC-only variant of Conformer: the encoder output is projected to vocabulary
size and decoded with CTC greedy search. Unlike the attention-decoder or transducer variants,
CTC decoding is fully non-autoregressive and very fast.

## Methods

| Method | Summary |
|:-----|:--------|
| `TokensToText(List<Int32>)` | Maps token IDs to text. |
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using Conformer encoder with CTC greedy decoding. |

