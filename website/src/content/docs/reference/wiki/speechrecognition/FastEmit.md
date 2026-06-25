---
title: "FastEmit<T>"
description: "FastEmit: low-latency RNN-Transducer with emission regularization"
section: "API Reference"
---

`Models & Types` · `AiDotNet.SpeechRecognition.Streaming`

FastEmit: low-latency RNN-Transducer with emission regularization

## For Beginners

FastEmit addresses the emission latency problem in RNN-Transducer models, where the model tends to delay token emissions. It introduces an emission regularization term that encourages the model to emit non-blank tokens as soon as possible without ...

## How It Works

**References:**

- Paper: "FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization" (Yu et al., Google, 2021)

FastEmit addresses the emission latency problem in RNN-Transducer models, where the model tends to delay token emissions. It introduces an emission regularization term that encourages the model to emit non-blank tokens as soon as possible without degrading accuracy. This sequence-level regularization modifies the RNN-T loss to penalize late emissions. FastEmit significantly reduces first-token and average emission latency while maintaining or improving word error rates.

## Methods

| Method | Summary |
|:-----|:--------|
| `Transcribe(Tensor<>,String,Boolean)` | Transcribes audio using FastEmit's emission-regularized RNN-T decoder. |

