---
title: "NeMoCitrinet<T>"
description: "NeMo Citrinet: 1D time-channel separable convolution CTC model with squeeze-and-excitation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.NeMo`

NeMo Citrinet: 1D time-channel separable convolution CTC model with squeeze-and-excitation.

## For Beginners

Citrinet extends QuartzNet/Jasper with: (1) 1D time-channel separable convolutions that factorize standard convolutions into depthwise temporal and pointwise channel components; (2) Squeeze-and-Excitation (SE) blocks for channel attention; (3) sub...

## How It Works

**References:**

- Paper: "Citrinet: Closing the Gap between Non-Autoregressive and Autoregressive End-to-End Models for Automatic Speech Recognition" (Majumdar et al., NVIDIA, 2021)

Citrinet extends QuartzNet/Jasper with: (1) 1D time-channel separable convolutions that
factorize standard convolutions into depthwise temporal and pointwise channel components;
(2) Squeeze-and-Excitation (SE) blocks for channel attention; (3) sub-word tokenization
instead of character-level. The architecture uses 5 blocks of repeated B-sub-blocks, each
containing 1D depthwise conv + pointwise conv + SE + residual. CTC decoding on sub-word
tokens achieves competitive WER with much faster inference than attention-based models.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using 1D time-channel separable convolutions with CTC decoding. |

